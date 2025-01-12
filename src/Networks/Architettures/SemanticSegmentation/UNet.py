from typing import Any, Final
from scipy.ndimage.filters import maximum_filter1d
import torch
import torch.nn as nn
from torch.nn import Module, Sequential

from DatasetComponents.DataModule.DataModuleBase import DataModuleBase
import Globals
from Networks.NetworkComponents.TrainingModel import Semantic_ImageSegmentation_TrainingBase
from ...NetworkComponents.NeuralNetworkBase import *


class _UnetBase(Semantic_ImageSegmentation_TrainingBase):
    
    _FEATURES: Final[Tuple[int, int, int, int]] = (64,128,256,512)
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        if kwargs.get('features') is None:
            self._features = self._FEATURES
        else:
            self._features = kwargs['features']
        
        
        #assert len(self._features) == 4, "The number of features must be 4"
        assert all(i > 0 for i in self._features), "The features must be positive integers"
        
        
        self._EncoderBlocks = nn.ModuleList()
        self._Bottleneck = nn.ModuleList()
        self._DecoderBlocks = nn.ModuleList()
        self._OutputLayer = nn.Sequential()
        
        
    def configure_lossFunction(self) -> nn.Module:
        
        if self._output_Classes == 1:
            return nn.BCELoss()
        else:
            Globals.APP_LOGGER.info(f"CrossEntropyLoss parametre:")
            Globals.APP_LOGGER.info(f"ignoreIndexFromLoss: {self._datamodule.getIgnoreIndexFromLoss}")
            Globals.APP_LOGGER.info(f"getWeights: {self._datamodule.getWeights}")
            
            # Ignora i pixel con classe "ignoreIndexFromLoss"
            return nn.CrossEntropyLoss(weight=self._datamodule.getWeights, ignore_index=self._datamodule.getIgnoreIndexFromLoss)
    
    
    def calculateLoss(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        #y_hat = self.__net(x)
        # print(x.dtype, x.shape)
        # print(y.dtype, y.shape)
        y_hat = self.forward(x)
        loss: float = 0.0
    
        if self._datamodule.getIgnoreIndexFromLoss >= 0 and self._datamodule.use_oneHot_encoding:
            y = y.argmax(dim=1)
            loss = self._lossFunction(y_hat, y)
        else:
            loss = self._lossFunction(y_hat, y.squeeze(1))
            
        return {"loss": loss, "y_hat": y_hat}
        
        
    def forward(self, x) -> torch.Tensor | None :
        
        skip_connections = []
        #print(x)

        for encoder_block in self._EncoderBlocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self._DownSampler(x)

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
        x = self._OutputLayer(x)
        # print(x.dtype, x.shape)
        # print(x)
        return x

class UNET_2D(_UnetBase):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        inputSize: list | None = self._datamodule.input_size
        inputSize = [1, self._in_Channel, 572,572] if inputSize is None else inputSize
        
        try:
            assert len(inputSize) == 4, "The input size must be a list of 4 elements"
        except AssertionError as e:
            print(f"Error: {e}")
            print("Input size: ", inputSize)
            raise
        
        
        self._DownSampler = nn.MaxPool2d(kernel_size=2, stride=2)
       
        
        in_feat = self._in_Channel
        
        for feature in self._features:
            self._EncoderBlocks.append(
                Multiple_Conv2D_Block(
                    num_convs=2,
                    in_channels=in_feat, 
                    out_channels=feature, 
                    kernel_size=(3,3), 
                    stride=(1,1), 
                    padding=(1,1),
                    bias=False
                )
            )
            in_feat = feature
            
        self._Bottleneck.append(
            Multiple_Conv2D_Block(
                num_convs=2,
                in_channels=self._features[-1],
                out_channels=self._features[-1]*2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )
        
        
        for feature in reversed(self._features):
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
                in_channels=self._features[0],
                out_channels=self._out_channel,
                kernel_size=1
            )
        )
        
        
        lossClass = self.configure_lossFunction()
        
        if isinstance(lossClass, nn.BCELoss):
            self._OutputLayer.append(nn.Sigmoid())   
        elif isinstance(lossClass, nn.CrossEntropyLoss):
            pass
        else:
            raise ValueError("The loss function is not supported")
        
    
    

        
        
        
        
        
       

class UNet_3D(_UnetBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        assert len(self._DataInputSize) == 5, "The input size must be a list of 5 elements"
        
        self._depth = self._DataInputSize[2]
        self._inputSize = [1, self._in_Channel, self._depth, 572,572]
        
      
        
        #================ DOWN_SAMPLER ================#
        
        self._DownSampler = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        

        #================ ENCODER ================#
        in_feat = self._in_Channel
        
        for feature in self._features:
            self._EncoderBlocks.append(
                Multiple_Conv3D_Block(
                    num_convs=2,
                    in_channels=in_feat, 
                    out_channels=feature, 
                    kernel_size=(3,3,3), 
                    stride=(1,1,1), 
                    padding=(1,1,1),
                    bias=True
                )
            )
            in_feat = feature
        
        #================ BRIDGE ================#
        self._Bottleneck.append(
            Multiple_Conv3D_Block(
                num_convs=2,
                in_channels=self._features[-1],
                out_channels=self._features[-1]*2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )
        #================ DECODER ================#
        for feature in reversed(self._features):
            self._DecoderBlocks.append(
                Deconv3D_Block(
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=(3,3,3),
                    stride=(2,2,2)
                )
            )
            self._DecoderBlocks.append(
                Multiple_Conv3D_Block(
                    num_convs=2,
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=(3,3,3),
                    stride=(1,1,1),
                    padding=(1,1,1),
                    bias=True
                )
            )
            
            
        self._OutputLayer = nn.Sequential(
            nn.Conv3d(
                in_channels=self._features[0],
                out_channels=1,
                kernel_size=(1,1,1),
                stride=(1,1,1),
                padding=0,
                bias=True
            ),
            Squeezer(1),
            nn.Conv2d(
                in_channels=self._depth, 
                out_channels=self._out_channel, 
                kernel_size=(1,1), 
                stride=(1,1), 
                padding=0, 
                bias=True
            )
        )
        
        
        lossClass = self.configure_lossFunction()
        
        if isinstance(lossClass, nn.BCELoss):
            self._OutputLayer.append(nn.Sigmoid())   
        elif isinstance(lossClass, nn.CrossEntropyLoss):
            pass
        else:
            raise ValueError("The loss function is not supported")