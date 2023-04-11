from typing import TYPE_CHECKING

import torch
from torch.nn import (
    Module,
    ModuleDict, 
    Conv2d, 
    ReLU, 
    MaxPool2d, 
    ModuleDict, 
    ConvTranspose2d,
    Sigmoid
)

if TYPE_CHECKING:
    from yaecs import Configuration

class AttentionLayer(Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        assert in_channels%2 == 0, "Attention layer: in_channels has to be even"
        self.conv1 = Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.reLU = ReLU()
        self.conv3 = Conv2d(in_channels//2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x, skip):
        x = self.conv1(x)
        skip = self.conv2(skip)
        attention_layer = self.reLU(x + skip)
        attention_layer = self.conv3(attention_layer)
        attention_layer = self.sigmoid(attention_layer)
        return attention_layer
    
class UNet(Module):
    
    def __init__(self, config: 'Configuration'):
        super().__init__()
        
        assert config.S1toS2_weights_path is not None, "Unet needs pretrained weights for the S1 to S2 autoencoder"
        
        S1toS2_autoEncoder = S1toS2AutoEncoder(config).float()
        S1toS2_autoEncoder.to(config.device)
        S1toS2_autoEncoder.load_state_dict(torch.load(config.S1toS2_weights_path))
        
        self.layers = ModuleDict({
            "S1toS2_autoencoder": S1toS2_autoEncoder,
            "S2_and_Mask_block": ModuleDict({
                "Conv_1": Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),
                "Conv_2": Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(),   
            }),
            "Encoder_block_1": ModuleDict({
                "Conv_1": Conv2d(in_channels=9, out_channels=32, kernel_size=3, padding='same', device=config.device),
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
                "Conv_1": Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_1": ReLU(),  
            }),
            "Attention_block_2": AttentionLayer(in_channels=64),
            "Decoder_block_2": ModuleDict({
                "Deconv_1": ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, device=config.device),
                "ReLU_1": ReLU(),
                "Conv_1": Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', device=config.device),
                "ReLU_2": ReLU(), 
                "Conv_2": Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same', device=config.device),
                "ReLU_3": ReLU(),   
            }),
            "Attention_block_3": AttentionLayer(in_channels=32),
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
        
        #freezing the top layers
        for param in S1toS2_autoEncoder.parameters():
            param.requires_grad = False
        
    def forward(self, S1_t0_data, S2_and_Mask_t2, S2_and_Mask_t1):
        x0 = self.layers["S1toS2_autoencoder"](S1_t0_data) 
        x1 = self.layers["S2_and_Mask_block"]["ReLU_1"](self.layers["S2_and_Mask_block"]["Conv_1"](S2_and_Mask_t1))
        x1 = self.layers["S2_and_Mask_block"]["ReLU_2"](self.layers["S2_and_Mask_block"]["Conv_2"](x1))
        x2 = self.layers["S2_and_Mask_block"]["ReLU_1"](self.layers["S2_and_Mask_block"]["Conv_1"](S2_and_Mask_t2))
        x2 = self.layers["S2_and_Mask_block"]["ReLU_2"](self.layers["S2_and_Mask_block"]["Conv_2"](x2))
        
        x = torch.cat([x0,x1,x2], dim=1)
        tensor_states = []
        #Encoder
        for encoder_block_index in range(1,3):
            encoder_bloc_name = f'Encoder_block_{encoder_block_index}'
            x = self.layers[encoder_bloc_name]["ReLU_1"](self.layers[encoder_bloc_name]["Conv_1"](x))
            x = self.layers[encoder_bloc_name]["ReLU_2"](self.layers[encoder_bloc_name]["Conv_2"](x))
            x = self.layers[encoder_bloc_name]["Max_pool"](x)
            tensor_states.append(x.clone())
        x = self.layers["Encoder_block_4"]["ReLU_1"](self.layers["Encoder_block_4"]["Conv_1"](x))
        
        #Decoder
        x = self.layers["Decoder_block_1"]["ReLU_1"](self.layers["Decoder_block_1"]["Conv_1"](x))   
        for decoder_block_index in range(2,4):
            decoder_bloc_name = f'Decoder_block_{decoder_block_index}'
            skip_connection = tensor_states.pop(-1)
            attention_layer = self.layers[f'Attention_block_{decoder_block_index}'](x, skip_connection)
            x = self.layers[decoder_bloc_name]["ReLU_1"](
                self.layers[decoder_bloc_name]["Deconv_1"]( torch.cat([x, torch.mul(attention_layer, skip_connection)], dim=1))
            )
            x = self.layers[decoder_bloc_name]["ReLU_2"](self.layers[decoder_bloc_name]["Conv_1"](x))
            x = self.layers[decoder_bloc_name]["ReLU_3"](self.layers[decoder_bloc_name]["Conv_2"](x))
        x = self.layers["Decoder_block_5"]["Conv_1"](x)

        return x

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
       
def get_model(config: 'Configuration') -> Module:
    """ Prepares the model to be used. This includes instantiation, weight loading and using eval mode or not. """
    if config.model == "S1ToS2Model":
        model = S1toS2AutoEncoder(config).float()
    elif  config.model == "Unet":
        model = UNet(config).float()
    else:
        raise ValueError(f"Unknown model '{config.model}'.")

    if config.weights_path is not None:
        model.to(config.device)
        model.load_state_dict(torch.load(config.weights_path))

    if config.mode != "train":
        model.eval()

    return model
