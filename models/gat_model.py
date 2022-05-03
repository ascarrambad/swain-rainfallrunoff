
from torch import nn

from tsl.utils.parser_utils import ArgParser, str_to_bool
from tsl.nn.blocks.encoders import ConditionalBlock
from .blocks.gat_block import SWAIN_SpatioTemporalConvNet
from tsl.nn.blocks.decoders.multi_step_mlp_decoder import MultiHorizonMLPDecoder


class SWAIN_GATModel(nn.Module):
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
            spatial_attention_heads (int): Number of heads to perform attention with.
            dilation (int, optional): Dilation of the temporal convolutional kernels.
            norm (str, optional): Normalization strategy.
            gated (bool, optional): Whether to use gated TanH activation in the temporal convolutional layers.
            activation (str, optional): Activation function.
            dropout (float, optional): Dropout probability.
        """
    def __init__(self,
                 use_node_attribs,
                 input_size,
                 exog_size,
                 model_hidden_size,
                 decoder_hidden_size,
                 decoder_context_size,
                 output_size,
                 n_layers,
                 horizon,
                 temporal_kernel_size,
                 spatial_attention_heads,
                 temporal_convs_layer=2,
                 spatial_convs_layer=1,
                 dilation=1,
                 norm='none',
                 gated=False,
                 activation='relu',
                 dropout=0.):
        super(SWAIN_GATModel, self).__init__()

        self.use_node_attribs = use_node_attribs

        if use_node_attribs == 'cond':
            self.node_cond = ConditionalBlock(input_size=input_size,
                                              exog_size=59,
                                              output_size=model_hidden_size,
                                              activation=activation)
        # elif use_node_attribs == 'ea':
        elif use_node_attribs != 'none':
            raise NotImplementedError(f'Usage "{use_node_attribs}" for node features not available.')


        self.exog_cond = ConditionalBlock(input_size=model_hidden_size if use_node_attribs != 'none' else input_size,
                                          exog_size=exog_size,
                                          output_size=model_hidden_size,
                                          activation=activation)

        conv_blocks = []
        for _ in range(n_layers):
            conv_blocks.append(
                SWAIN_SpatioTemporalConvNet(
                    input_size=model_hidden_size,
                    output_size=model_hidden_size,
                    temporal_kernel_size=temporal_kernel_size,
                    spatial_attention_heads=spatial_attention_heads,
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

        # self.readout = MultiHorizonMLPDecoder(input_size=model_hidden_size,
        #                                       exog_size=exog_size,
        #                                       hidden_size=decoder_hidden_size,
        #                                       context_size=decoder_context_size,
        #                                       output_size=output_size,
        #                                       n_layers=n_layers,
        #                                       horizon=horizon,
        #                                       activation=activation,
        #                                       dropout=dropout)
        self.readout = ConditionalBlock(input_size=model_hidden_size,
                                        exog_size=exog_size,
                                        output_size=output_size,
                                        activation=activation)

    def forward(self, x, u_w, u_h, edge_index, node_attr=None, edge_attr=None, edge_weight=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, steps, nodes, channels]
        if self.use_node_attribs == 'cond':
            x = self.node_cond(x, node_attr)
        x = self.exog_cond(x, u_w)

        for conv in self.convs:
            x = x + conv(x, edge_index, edge_attr)

        return self.readout(x, u_h)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--use-node-attribs', type=str, default='none', tunable=True, options=['none', 'cond', 'ea'])
        parser.opt_list('--model-hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128])
        parser.opt_list('--decoder-hidden-size', type=int, default=256, tunable=True, options=[64, 128, 256, 512])
        parser.opt_list('--decoder-context-size', type=int, default=256, tunable=True, options=[16, 32, 64, 128])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True, options=[1, 2])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--temporal-kernel-size', type=int, default=2, tunable=True, options=[2, 3, 5])
        parser.opt_list('--spatial-attention-heads', type=int, default=1, tunable=True, options=[1, 3, 5])
        parser.opt_list('--dilation', type=int, default=2, tunable=True, options=[1, 2])
        parser.opt_list('--norm', type=str, default='none', tunable=True, options=['none', 'layer', 'batch'])
        parser.opt_list('--gated', type=str_to_bool, tunable=False, nargs='?', const=True, default=False, options=[True, False])
        return parser