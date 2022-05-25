from torch import nn

from tsl.nn.layers.graph_convs.gat_conv import GATConv
from tsl.nn.blocks.encoders.tcn import TemporalConvNet
from tsl.nn.layers.norm.norm import Norm

from tsl.nn.utils import utils


class SWAIN_SpatioTemporalConvNet(nn.Module):
    r"""
        SpatioTemporalConvolutional encoder with optional linear readout.
        Applies several temporal convolutions followed by diffusion convolution over a graph.

        Args:
            input_size (int): Input size.
            output_size (int): Channels in the output representation.
            temporal_kernel_size (int): Size of the temporal convolutional kernel.
            spatial_kernel_size (int): Size of the spatial diffusion kernel.
            temporal_convs (int, optional): Number of temporal convolutions. (default: 2)
            spatial_convs (int, optional): Number of spatial convolutions. (default: 1)
            dilation (int): Dilation coefficient of the temporal convolutional kernel.
            norm (str, optional): Type of normalization applied to the hidden units.
            dropout (float, optional): Dropout probability.
            gated (bool, optional): Whether to used the GatedTanH activation function after temporal convolutions.
                                    (default: `False`)
            pad (bool, optional): Whether to pad the input sequence to preserve the sequence length.
            activation (str, optional): Activation function. (default: `relu`)
        """
    def __init__(self,
                 input_size,
                 output_size,
                 temporal_kernel_size,
                 spatial_attention_heads,
                 temporal_convs=2,
                 spatial_convs=1,
                 dilation=1,
                 norm='none',
                 dropout=0.,
                 gated=False,
                 pad=True,
                 activation='relu'):
        super(SWAIN_SpatioTemporalConvNet, self).__init__()
        self.pad = pad

        self.tcn = nn.Sequential(
            Norm(norm_type=norm, in_channels=input_size),
            TemporalConvNet(
                input_channels=input_size,
                hidden_channels=output_size,
                kernel_size=temporal_kernel_size,
                dilation=dilation,
                exponential_dilation=True,
                n_layers=temporal_convs,
                activation=activation,
                causal_padding=pad,
                dropout=dropout,
                gated=gated
            ))

        self.skip_conn = nn.Linear(input_size, output_size)

        self.spatial_convs = nn.ModuleList(GATConv(in_channels=output_size,
                                                   out_channels=output_size,
                                                   heads=spatial_attention_heads,
                                                   concat=False,
                                                   dropout=dropout,
                                                   add_self_loops=False,
                                                   edge_dim=3) for _ in range(spatial_convs))
        self.spatial_norms = nn.ModuleList(Norm(norm_type=norm, in_channels=output_size)
                                           for _ in range(spatial_convs))
        self.dropout = nn.Dropout(dropout)
        self.activation = utils.get_functional_activation(activation)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        # temporal conv
        x = self.skip_conn(x) + self.tcn(x)
        # spatial conv
        for filter, norm in zip(self.spatial_convs, self.spatial_norms):
            x, _ = filter(norm(x), edge_index, edge_attr)
            x = x + self.dropout(self.activation(x))
        return x
