from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Conv2d
if TYPE_CHECKING:
    from yaecs import Configuration


class BaselineModel(Module):
    """ A very basic model to serve as a baseline. """
    def __init__(self, config: 'Configuration'):
        super().__init__()
        last_hidden_layer_channels = 8

        self.hidden_layers = []
        for channel in config.layers_channels:
            self.hidden_layers.append(Conv2d(in_channels=last_hidden_layer_channels, out_channels=channel,
                                             kernel_size=(3, 3), padding='same', device=config.device))
            last_hidden_layer_channels = channel

        self.output = Conv2d(in_channels=last_hidden_layer_channels, out_channels=1,
                             kernel_size=(3, 3), padding='same', device=config.device)

    def forward(self, sample):
        x = torch.cat([sample["s1"], sample["s2"]], dim=-1).permute(0, 3, 1, 2).float()
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output(x).permute(0, 2, 3, 1)


def get_model(config: 'Configuration') -> BaselineModel:
    """ Prepares the model to be used. This includes instantiation, weight loading and using eval mode or not. """
    model = BaselineModel(config).float()

    if config.weights_path is not None:
        model.to(config.device)
        model.load_state_dict(torch.load(config.weights_path))

    if config.mode != "train":
        model.eval()

    return model
