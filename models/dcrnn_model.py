
from torch import nn

from tsl.nn.blocks.encoders.dcrnn import DCRNN
from tsl.nn.blocks.encoders import ConditionalBlock

from tsl.nn.blocks.decoders.multi_step_mlp_decoder import MultiHorizonMLPDecoder

from tsl.nn.models.stgn import DCRNNModel


class SWAIN_DCRNNModel(DCRNNModel):
    r"""
    Diffusion ConvolutionalRecurrent Neural Network with a nonlinear readout.

    From Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the DCRNN hidden layer.
        ff_size (int): Number of units in the nonlinear readout.
        output_size (int): Number of output channels.
        n_layers (int): Number DCRNN cells.
        exog_size (int): Number of channels in the exogenous variable.
        horizon (int): Number of steps to forecast.
        activation (str, optional): Activation function in the readout.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 output_size,
                 n_layers,
                 exog_size,
                 horizon,
                 activation='relu',
                 dropout=0.,
                 kernel_size=2):
        super(DCRNNModel, self).__init__()

        self.input_encoder = ConditionalBlock(input_size=input_size,
                                              exog_size=exog_size,
                                              output_size=hidden_size,
                                              activation=activation)

        self.dcrnn = DCRNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           k=kernel_size)

        self.readout = MultiHorizonMLPDecoder(input_size=hidden_size,
                                              exog_size=exog_size,
                                              hidden_size=ff_size,
                                              context_size=hidden_size,
                                              output_size=output_size,
                                              n_layers=n_layers,
                                              horizon=horizon,
                                              activation=activation,
                                              dropout=dropout)


    def forward(self, x, edge_index, u_w, u_h, edge_weight=None, **kwargs):
        x = self.input_encoder(x, u_w)
        h, _ = self.dcrnn(x, edge_index, edge_weight)

        return self.readout(h, u_h)
