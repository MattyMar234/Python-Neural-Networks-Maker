import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import lightning as L
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
    
    
class LightModelBase(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super(*args, **kwargs).__init__()
        
    def makeSummary(self, depth: int = 4) -> str:
        colName = ['input_size', 'output_size', 'num_params', 'trainable']
        temp = summary(self, input_size=self.requestedInputSize(), col_width=20, col_names=colName, row_settings=['var_names'], verbose=0, depth=depth)
        return temp.__repr__()
        
    @abstractmethod
    def requestedInputSize(self) -> list[int]:
        pass




    
    
    



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