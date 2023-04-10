from typing import TYPE_CHECKING

import torch
from torch.nn import (
    Module,
    ModuleDict, 
    Conv2d, 
    ReLU, 
    Linear,
    MaxPool2d, 
    ModuleDict, 
    ConvTranspose2d
)

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

class S1toS2AutoEncoder(Module):
    
    def __init__(self, config: 'Configuration'):
        super().__init__()
        
        self.layers = ModuleDict({
            "Encoder_block_1": ModuleDict({
                "Conv_1": Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),
                "Conv_2": Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(), 
                "Max_pool": MaxPool2d(kernel_size=2, stride=2)    
            }),
            "Encoder_block_2": ModuleDict({
                "Conv_1": Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),
                "Conv_2": Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(), 
                "Max_pool": MaxPool2d(kernel_size=2, stride=2)    
            }),
            "Encoder_block_4": ModuleDict({
                "Conv_1": Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),   
            }),
            "Decoder_block_1": ModuleDict({
                "Conv_1": Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),  
            }),
            "Decoder_block_2": ModuleDict({
                "Deconv_1": ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, device=config.device),
                "ReLU_1": ReLU(),
                "Conv_1": Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(), 
                "Conv_2": Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_3": ReLU(),   
            }),
            "Decoder_block_3": ModuleDict({
                "Deconv_1": ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, device=config.device),
                "ReLU_1": ReLU(),
                "Conv_1": Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(), 
                "Conv_2": Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same', device=config.device),
                "ReLU_3": ReLU(),  
            }),
            "Decoder_block_5": ModuleDict({
                "Conv_1": Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same', device=config.device)
            }),
                
        })  
        
    def forward(self, sample):
        x = sample.clone()
        
        #Encoder
        for encoder_block_index in range(1,3):
            encoder_bloc_name = f'Encoder_block_{encoder_block_index}'
            x = self.layers[encoder_bloc_name]["ReLU_1"](self.layers[encoder_bloc_name]["Conv_1"](x))
            x = self.layers[encoder_bloc_name]["ReLU_2"](self.layers[encoder_bloc_name]["Conv_2"](x))
            x = self.layers[encoder_bloc_name]["Max_pool"](x)
        x = self.layers["Encoder_block_4"]["ReLU_1"](self.layers["Encoder_block_4"]["Conv_1"](x))
        
        #Decoder
        x = self.layers["Decoder_block_1"]["ReLU_1"](self.layers["Decoder_block_1"]["Conv_1"](x))   
        for decoder_block_index in range(2,4):
            decoder_bloc_name = f'Decoder_block_{decoder_block_index}'
            x = self.layers[decoder_bloc_name]["ReLU_1"](self.layers[decoder_bloc_name]["Deconv_1"](x))
            x = self.layers[decoder_bloc_name]["ReLU_2"](self.layers[decoder_bloc_name]["Conv_1"](x))
            x = self.layers[decoder_bloc_name]["ReLU_3"](self.layers[decoder_bloc_name]["Conv_2"](x))
        x = self.layers["Decoder_block_5"]["Conv_1"](x)

        return x
       
def get_model(config: 'Configuration') -> BaselineModel:
    """ Prepares the model to be used. This includes instantiation, weight loading and using eval mode or not. """
    if config.model == "S1ToS2Model":
        model = S1toS2AutoEncoder(config).float()
    else:
        model = BaselineModel(config).float()


    if config.weights_path is not None:
        model.to(config.device)
        model.load_state_dict(torch.load(config.weights_path))

    if config.mode != "train":
        model.eval()

    return model
