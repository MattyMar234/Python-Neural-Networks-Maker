from typing import Any, Final
from scipy.ndimage.filters import maximum_filter1d
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from scipy.ndimage.filters import maximum_filter1d
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from DatasetComponents.DataModule.DataModuleBase import DataModuleBase
import Globals
from Networks.NetworkComponents.TrainingModel import Semantic_ImageSegmentation_TrainingBase
from ...NetworkComponents.NeuralNetworkBase import *


class _UnetBase(Semantic_ImageSegmentation_TrainingBase):
    
    _FEATURES: Final[Tuple[int, int, int, int]] = (64,128,256,512)
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # if kwargs.get('features') is None:
        #     self._features = self._FEATURES
        # else:
        #     self._features = kwargs['features']
        
        self._features = self._FEATURES
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

        return self._OutputLayer(x)

class UNet_3D_v2(Semantic_ImageSegmentation_TrainingBase):

    #def __init__(self, depth, in_channels, out_classes, feat_channels=[48, 256, 256, 512, 1024], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        #super(UNet_3D, self).__init__()
        
        feat_channels=[48, 256, 256, 512, 1024]
        residual='conv'

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        in_channels = self._in_Channel 
        depth = self._DataInputSize[2]
        out_classes = self._output_Classes

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block2(
            in_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block2(
            feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block2(
            feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block2(
            feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block2(
            feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block2(
            2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block2(
            2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block2(
            2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block2(
            2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block2(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block2(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block2(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block2(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(
            feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.final_conv = torch.nn.Conv2d(depth, out_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = nn.Sigmoid()

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
        y_hat = self.forward(x)
        loss: float = 0.0
    
        if self._datamodule.getIgnoreIndexFromLoss >= 0 and self._datamodule.use_oneHot_encoding:
            y = y.argmax(dim=1)
            loss = self._lossFunction(y_hat, y)
        else:
            loss = self._lossFunction(y_hat, y.squeeze(1))
            
        return {"loss": loss, "y_hat": y_hat}

    def forward(self, x):
        # Encoder part

        print(x.shape)

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)
        r = self.deconv_blk4(base)
        # Decoder part
        print(x4.shape, r.shape)
        
        d4 = torch.cat([r, x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        test2 = self.deconv_blk3(d_high4)

        d3 = torch.cat([test2, x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        # print(x4.shape, x3.shape, x2.shape, x1.shape)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        conv_out = self.one_conv(d_high1)
        conv_out = conv_out.squeeze(1)
        out = self.final_conv(conv_out)
        # seg = self.sigmoid(out)
        return out

class Deconv3D_Block2(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block2, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            nn.ReLU())

    def forward(self, x):
        return self.deconv(x)

class Conv3D_Block2(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block2, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            nn.ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            nn.ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(
                inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(
            self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)



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