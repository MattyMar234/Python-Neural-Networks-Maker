import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from abc import ABC, abstractmethod
from enum import IntEnum, auto

from torchinfo import summary
from Networks.NetworkComponents.NeuralNetworkBase import *

        


    

class LaNet5_ReLU(ModelBase):
    def __init__(self, in_channel, num_classes=10):
        super().__init__(num_classes)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
    
    
class LaNet5_TanH(ModelBase):
    def __init__(self, in_channel: int, num_classes: int):
        super(num_classes).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))
    
    
    
class AlexNet(ModelBase):
    def __init__(self, in_channel: int, num_classes: int):
        super(num_classes).__init__()
    
        self.__size = [1, in_channel, 224,224]
        
        self.features = nn.Sequential(
            Conv2dBlock(
                pool=PoolType.MAX, in_channels=in_channel, out_channels = 96,
                conv_kernel_size=11, conv_kernel_stride=4, conv_kernel_padding=0, 
                pool_kernel_Size=3, pool_kernel_stride=2
            ),
            
            Conv2dBlock(
                PoolType.MAX, in_channels=96, out_channels = 256,
                conv_kernel_size=5, conv_kernel_stride=1, conv_kernel_padding=2,
                pool_kernel_Size=3, pool_kernel_stride=2
            ),
            Conv2dBlock(
                PoolType.NONE, in_channels=256, out_channels = 384,
                conv_kernel_size=3, conv_kernel_stride=1, conv_kernel_padding=1,
            ),
            Conv2dBlock(
                PoolType.NONE, in_channels=384, out_channels = 384,
                conv_kernel_size=3, conv_kernel_stride=1, conv_kernel_padding=1,
            ),
            Conv2dBlock(
                PoolType.MAX, in_channels=384, out_channels = 256,
                conv_kernel_size=3, conv_kernel_stride=1, conv_kernel_padding=1,
                pool_kernel_Size=3, pool_kernel_stride=2
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )
        
        self.net = nn.Sequential(self.features, self.classifier)

    def forward(self, x):
       return self.net(x)
   
    def requestedInputSize(self) -> list[int]:
        return self.__size
   
   
class VGG_19(ModelBase):
    
    __max_classes = 1000
    
    class VGG_19_Block(nn.Module):
        def __init__(self, num_convs, in_channels, out_channels):
            super().__init__()
            
            self.blockComponents = nn.Sequential()
            
            for i in range(num_convs):
                
                if(i == 0):
                    b = Conv2dBlock(
                        PoolType.NONE, in_channels=in_channels, out_channels=out_channels,
                        conv_kernel_size=3, conv_kernel_stride=1, conv_kernel_padding=1,
                    )
                else:
                    b = Conv2dBlock(
                        PoolType.NONE, in_channels=out_channels, out_channels=out_channels,
                        conv_kernel_size=3, conv_kernel_stride=1, conv_kernel_padding=1,
                    )
                
                self.blockComponents.append(b)
                
            self.blockComponents.append(
                nn.MaxPool2d(kernel_size=2, stride=2) 
            )
                
                
            
        def forward(self, x):
            return self.blockComponents(x)
    
    def __init__(self, in_channel: int, num_classes: int):
        assert num_classes <= VGG_19.__max_classes
        
        super(num_classes).__init__()
        self.__size = [1, in_channel, 224,224]

        self.features = nn.Sequential(
            VGG_19.VGG_19_Block(num_convs=2, in_channels=in_channel, out_channels=64),
            VGG_19.VGG_19_Block(num_convs=2, in_channels=64, out_channels=128),
            VGG_19.VGG_19_Block(num_convs=4, in_channels=128, out_channels=256),
            VGG_19.VGG_19_Block(num_convs=4, in_channels=256, out_channels=512),
            VGG_19.VGG_19_Block(num_convs=4, in_channels=512, out_channels=512)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )
        
        self.net = nn.Sequential(self.features, self.classifier)
        #self.net = nn.Sequential(self.features)
    
    
    def forward(self, x):
       return self.net(x)
   
    def requestedInputSize(self) -> list[int]:
        return self.__size