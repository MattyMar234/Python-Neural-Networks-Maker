from typing import List, Optional, Tuple
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchinfo import summary

from abc import ABC, abstractmethod
from enum import Enum



class PoolType(Enum):
    NONE = 0
    MAX = 1
    AVG = 2


class ModelBase(nn.Module):
    def __init__(self, classes: int) -> None:
        super(ModelBase, self).__init__()
        
        self._classes = classes
        
        
        
    def makeSummary(self, depth: int = 4) -> str:
        colName = ['input_size', 'output_size', 'num_params', 'trainable']
        temp = summary(self, input_size=self.requestedInputSize(), col_width=20, col_names=colName, row_settings=['var_names'], verbose=0, depth=depth)
        return temp.__repr__()
    
    @property
    def outputClasses(self) -> int:
       return self._classes
        
    @abstractmethod
    def requestedInputSize(self) -> list[int]:
        pass
    
    
class LightModelBase(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self._datamodule = kwargs.get("datamodule")
        self._in_Channel:int = self._datamodule.input_channels
        self._out_channel: int = self._datamodule.output_classes
        self._output_Classes: int = self._datamodule.output_classes
        self._DataInputSize: Optional[List[int]] = self._datamodule.input_size
        self._inputSize: Optional[List[int]] = None
        
        assert self._in_Channel > 0, "Input channel must be greater than 0"
        assert self._out_channel > 0, "Output channel must be greater than 0"
        assert self._output_Classes > 0, "Output classes must be greater than 0"
        assert self._DataInputSize is not None, "Input size must be not None"

        #self.__net = kwargs['net']
        
        #print(self.__net)
        #self.save_hyperparameters()
        
        
    def makeSummary(self, depth: int = 4) -> str:
        colName = ['input_size', 'output_size', 'num_params', 'trainable']
        temp = summary(self, input_size=self._DataInputSize, col_width=20, col_names=colName, row_settings=['var_names'], verbose=0, depth=depth)
        return temp.__repr__()
        
    




    
    
    



class Conv2dBlock(nn.Module):
    
    def __init__(self, pool: PoolType = PoolType.NONE, in_channels: int = 1, out_channels: int = 1, conv_kernel_size: int = 3, conv_kernel_stride: int = 1, conv_kernel_padding: int = 1, pool_kernel_Size: int = 2, pool_kernel_stride: int = 2 ):
        super(Conv2dBlock, self).__init__()
        
        
        self.convolutionBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=conv_kernel_stride, padding=conv_kernel_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        match pool:
            
            case PoolType.NONE:
                pass
            
            case PoolType.MAX:
                self.convolutionBlock.append(
                    nn.MaxPool2d(kernel_size=pool_kernel_Size, stride=pool_kernel_stride)
                )
            case PoolType.AVG:
                self.convolutionBlock.append(
                    nn.AvgPool2d(kernel_size=pool_kernel_Size, stride=pool_kernel_stride)
                )
                
    def forward(self, x):
        return self.convolutionBlock(x)
    
    

class Multiple_Conv2D_Block(nn.Module):
        def __init__(self, num_convs: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool):
            super().__init__()
            
            self.blockComponents = nn.Sequential()
            
            for _ in range(num_convs):  
                self.blockComponents.append(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding, 
                        bias=bias
                    )
                )
                       
                in_channels = out_channels
                self.blockComponents.append(nn.BatchNorm2d(out_channels))
                self.blockComponents.append(nn.ReLU(inplace=False))
                
        def forward(self, x):
            return self.blockComponents(x)
        
        
class Multiple_Conv3D_Block(nn.Module):
        def __init__(
            self, 
            num_convs: int, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int | Tuple[int, int, int], 
            stride: int | Tuple[int, int, int] = 1, 
            padding: int | Tuple[int, int, int] = 0, 
            bias: bool = False
        ):
            
            assert isinstance(num_convs, int), "num_convs must be an integer"
            assert num_convs > 0, "num_convs must be greater than 0"
            assert isinstance(in_channels, int), "in_channels must be an integer"
            assert in_channels > 0, "in_channels must be greater than 0"
            assert isinstance(out_channels, int), "out_channels must be an integer"
            assert out_channels > 0, "out_channels must be greater than 0"
            assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "kernel_size must be an integer or a tuple"
            assert isinstance(stride, int) or isinstance(stride, tuple), "stride must be an integer or a tuple"
            assert isinstance(padding, int) or isinstance(padding, tuple), "padding must be an integer or a tuple"
            
            super().__init__()
            
            self.blockComponents = nn.Sequential()
            
            for _ in range(num_convs):  
                self.blockComponents.append(
                    nn.Conv3d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding, 
                        bias=bias
                    )
                )
                       
                in_channels = out_channels
                self.blockComponents.append(nn.BatchNorm3d(out_channels))
                self.blockComponents.append(nn.ReLU(inplace=True))
                
        def forward(self, x):
            return self.blockComponents(x)
        

class Deconv3D_Block(nn.Module) :
    
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int | Tuple[int, int, int], 
            stride: int | Tuple[int, int, int] = 1, 
            padding: int | Tuple[int, int, int] = 0, 
            bias: bool = False
        ):
        
        
        assert isinstance(in_channels, int), "in_channels must be an integer"
        assert in_channels > 0, "in_channels must be greater than 0"
        assert isinstance(out_channels, int), "out_channels must be an integer"
        assert out_channels > 0, "out_channels must be greater than 0"
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "kernel_size must be an integer or a tuple"
        assert isinstance(stride, int) or isinstance(stride, tuple), "stride must be an integer or a tuple"
        assert isinstance(padding, int) or isinstance(padding, tuple), "padding must be an integer or a tuple"
        
        super(Deconv3D_Block, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding, 
                output_padding=1, 
                bias=bias
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)
