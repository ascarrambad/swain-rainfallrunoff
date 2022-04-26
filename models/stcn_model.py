
from torch import nn

from tsl.nn.blocks.encoders import ConditionalBlock
from .blocks.stcn_block import SWAIN_SpatioTemporalConvNet

from tsl.nn.blocks.decoders.multi_step_mlp_decoder import MultiHorizonMLPDecoder

from tsl.nn.models.stgn import STCNModel


class SWAIN_STCNModel(STCNModel):
    r"""
        Spatiotemporal GNN with interleaved temporal and spatial diffusion convolutions.
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
            dilation (int, optional): Dilation of the temporal convolutional kernels.
            norm (str, optional): Normalization strategy.
            gated (bool, optional): Whether to use gated TanH activation in the temporal convolutional layers.
            activation (str, optional): Activation function.
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
                 temporal_convs_layer=2,
                 spatial_convs_layer=1,
                 dilation=1,
                 norm='none',
                 gated=False,
                 activation='relu',
                 dropout=0.):
        super(STCNModel, self).__init__()

        self.input_encoder = ConditionalBlock(input_size=input_size,
                                              exog_size=exog_size,
                                              output_size=hidden_size,
                                              activation=activation)

        conv_blocks = []
        for _ in range(n_layers):
            conv_blocks.append(
                SWAIN_SpatioTemporalConvNet(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    temporal_kernel_size=temporal_kernel_size,
                    spatial_kernel_size=spatial_kernel_size,
                    temporal_convs=temporal_convs_layer,
                    spatial_convs=spatial_convs_layer,
                    dilation=dilation,
                    norm=norm,
                    dropout=dropout,
                    gated=gated,
                    activation=activation
                )
            )
        self.convs = nn.ModuleList(conv_blocks)

        self.readout = MultiHorizonMLPDecoder(input_size=hidden_size,
                                              exog_size=exog_size,
                                              hidden_size=ff_size,
                                              context_size=hidden_size,
                                              output_size=output_size,
                                              n_layers=n_layers,
                                              horizon=horizon,
                                              activation=activation,
                                              dropout=dropout)

    def forward(self, x, u_w, u_h, edge_index, node_attr=None, edge_attr=None, edge_weight=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, steps, nodes, channels]
        x = self.input_encoder(x, u_w)

        for conv in self.convs:
            x = x + conv(x, edge_index, edge_attr)

        return self.readout(x, u_h)