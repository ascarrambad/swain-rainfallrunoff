from torch import nn
from torch.nn import functional as F
import torch

from tsl.nn.blocks.encoders.tcn import TemporalConvNet
from tsl.nn.base.embedding import StaticGraphEmbedding
from tsl.nn.blocks.decoders.multi_step_mlp_decoder import MultiHorizonMLPDecoder
from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.layers.graph_convs.dense_spatial_conv import SpatialConvOrderK
from tsl.nn.layers.norm.norm import Norm

from tsl.nn.models.stgn import GraphWaveNetModel


class SWAIN_GraphWaveNetModel(GraphWaveNetModel):
    r"""
    Graph WaveNet Model from Wu et al., ”Graph WaveNet for Deep Spatial-Temporal Graph Modeling”, IJCAI 2019
    Args:
        input_size (int): Size of the input.
        exog_size (int): Size of the exogenous variables.
        hidden_size (int): Number of units in the hidden layer.
        ff_size (int): Number of units in the hidden layers of the nonlinear readout.
        output_size (int): Number of output channels.
        n_layers (int): Number of GraphWaveNet blocks.
        horizon (int): Forecasting horizon.
        temporal_kernel_size (int): Size of the temporal convolution kernel.
        spatial_kernel_size (int): Order of the spatial diffusion process.
        learned_adjacency (bool): Whether to consider an additional learned adjacency matrix.
        n_nodes (int, optional): Number of nodes in the input graph. Only needed if `learned_adjacency` is `True`.
        emb_size (int, optional): Number of features in the node embeddings used for graph learning.
        dilation (int, optional): Dilation of the temporal convolutional kernels.
        dilation_mod (int, optional): Length of the cycle for the dilation coefficient.
        norm (str, optional): Normalization strategy.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 exog_size,
                 hidden_size,
                 ff_size,
                 output_size,
                 n_layers,
                 horizon,
                 temporal_kernel_size,
                 spatial_kernel_size,
                 learned_adjacency,
                 n_nodes=None,
                 emb_size=8,
                 dilation=2,
                 dilation_mod=2,
                 norm='batch',
                 dropout=0.):
        super(GraphWaveNetModel, self).__init__()

        if learned_adjacency:
            assert n_nodes is not None
            self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
            self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
        else:
            self.register_parameter('source_embedding', None)
            self.register_parameter('target_embedding', None)

        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        temporal_conv_blocks = []
        spatial_convs = []
        skip_connections = []
        norms = []
        receptive_field = 1
        for i in range(n_layers):
            d = dilation ** (i % dilation_mod)
            temporal_conv_blocks.append(TemporalConvNet(
                input_channels=hidden_size,
                hidden_channels=hidden_size,
                kernel_size=temporal_kernel_size,
                dilation=d,
                exponential_dilation=False,
                n_layers=1,
                causal_padding=False,
                gated=True
            )
            )

            spatial_convs.append(DiffConv(in_channels=hidden_size,
                                          out_channels=hidden_size,
                                          k=spatial_kernel_size))

            skip_connections.append(nn.Linear(hidden_size, ff_size))
            norms.append(Norm(norm, hidden_size))
            receptive_field += d * (temporal_kernel_size - 1)
        self.tconvs = nn.ModuleList(temporal_conv_blocks)
        self.sconvs = nn.ModuleList(spatial_convs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        self.receptive_field = receptive_field

        dense_sconvs = []
        if learned_adjacency:
            for _ in range(n_layers):
                dense_sconvs.append(
                    SpatialConvOrderK(input_size=hidden_size,
                                      output_size=hidden_size,
                                      support_len=1,
                                      order=spatial_kernel_size,
                                      include_self=False,
                                      channel_last=True)
                )
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.dense_sconvs_act = nn.ReLU()
        self.readout = MultiHorizonMLPDecoder(input_size=ff_size,
                                              exog_size=exog_size,
                                              hidden_size=2 * ff_size,
                                              context_size=2 * ff_size,
                                              output_size=output_size,
                                              n_layers=n_layers,
                                              horizon=horizon,
                                              activation='relu',
                                              dropout=dropout)

    def forward(self, x, u_w, u_h, edge_index, edge_weight=None, **kwargs):
        """"""
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]

        x = torch.cat([x, u_w], -1)

        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()

        x = self.input_encoder(x)

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (tconv, sconv, skip_conn, norm) in enumerate(
                zip(self.tconvs, self.sconvs, self.skip_connections, self.norms)):
            res = x
            # temporal conv
            x = tconv(x)
            # residual connection -> out
            out = skip_conn(x) + out[:, -x.size(1):]
            # spatial conv
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
            x = self.dropout(x)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)

        out = self.dense_sconvs_act(out)
        return self.readout(out, u_h)