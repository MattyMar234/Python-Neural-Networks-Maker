from typing import Any
from scipy.ndimage.filters import maximum_filter1d
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from .NeuralNetworkBase import *

class UNET_2D(LightModelBase):
    def __init__(self, in_Channel:int = 1, out_channel: int = 1, features: tuple[int, int, int, int, int] = (64,128,256,512)) -> None:
        super().__init__()
        
        assert len(features) == 4, "The number of features must be 5"
        assert all(i > 0 for i in features), "The features must be positive integers"
        

        self._in_Channel = in_Channel
        self._out_Channel = out_channel
        self._features = features
        
        self._EncoderBlocks = nn.ModuleList()
        self._Bottleneck = nn.ModuleList()
        self._DecoderBlocks = nn.ModuleList()
        self._OutputLayer = nn.ModuleList()
        self._Pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self._EncoderBlocks.append(
                Multiple_Conv2D_Block(
                    num_convs=2,
                    in_channels=in_Channel, 
                    out_channels=feature, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    bias=False
                )
            )
            in_Channel = feature
            
        self._Bottleneck.append(
            Multiple_Conv2D_Block(
                num_convs=2,
                in_channels=features[-1],
                out_channels=features[-1]*2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        
        
        for feature in reversed(features):
            self._DecoderBlocks.append(
                nn.ConvTranspose2d(
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self._DecoderBlocks.append(
                Multiple_Conv2D_Block(
                    num_convs=2,
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            
        self._OutputLayer.append(
            nn.Conv2d(
                in_channels=features[0],
                out_channels=out_channel,
                kernel_size=1
            )
        )
    
    def forward(self, x) -> torch.Tensor | None :
        skip_connections = []

        for encoder_block in self._EncoderBlocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self._Pool(x)

        x = self._Bottleneck[0](x)

        #revers della lista 
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self._DecoderBlocks), 2):
            
            #ConvTranspose2d sul risultato precedente
            x = self._DecoderBlocks[i](x)
            
            #Ottengo la copia del tensore
            skip_connection = skip_connections[int(i//2)]
            
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            #Concateno i tensori
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            #Eseguo le due convoluzioni 
            x = self._DecoderBlocks[i+1](concat_skip)

        return self._OutputLayer[0](x)

        
        
        
        
        
       

class UNet_3D(LightModelBase):
    pass
