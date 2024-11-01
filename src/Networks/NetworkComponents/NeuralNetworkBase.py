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
        
        self._in_Channel:int = kwargs['in_channel']
        self._out_channel: int = kwargs['out_channel']
        self._output_Classes: int = kwargs['output_Classes']
        self._inputSize: list = kwargs['inputSize']
        
        assert self._in_Channel > 0, "Input channel must be greater than 0"
        assert self._out_channel > 0, "Output channel must be greater than 0"
        assert self._output_Classes > 0, "Output classes must be greater than 0"

        #self.__net = kwargs['net']
        
        #print(self.__net)
        #self.save_hyperparameters()
        
        
    def makeSummary(self, depth: int = 4) -> str:
        colName = ['input_size', 'output_size', 'num_params', 'trainable']
        temp = summary(self, input_size=self._inputSize, col_width=20, col_names=colName, row_settings=['var_names'], verbose=0, depth=depth)
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