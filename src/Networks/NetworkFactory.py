import torch.nn as nn

from enum import IntEnum, auto
from Networks.NetworkComponents.NetworkModels import *
from Networks.TrainingModel import *

    


class NetworkFactory:
    
    class NetworkType(IntEnum):
        LaNet5_ReLU = auto(),
        LaNet5_TanH = auto(),
        AlexNet = auto(),
        VGG_19 = auto(),
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def makeNetwork(trainingModel: TrainModel_Type, networkType: NetworkType, in_channel: int, num_classes: int, timeSequenze: int = 1, trainer_lr: float = 1e-2) -> tuple[nn.Module, L.LightningModule]:
        
        net: nn.Module = None
        trainer: L.LightningModule = None
        
        match networkType:
            case NetworkFactory.NetworkType.LaNet5_ReLU:
                net = LaNet5_ReLU(in_channel=in_channel, num_classes=num_classes)
            
            case NetworkFactory.NetworkType.LaNet5_TanH:
                net = LaNet5_TanH(in_channel=in_channel, num_classes=num_classes)
            
            case NetworkFactory.NetworkType.AlexNet:
                net =  AlexNet(in_channel=in_channel*timeSequenze, num_classes=num_classes)
            
            case NetworkFactory.NetworkType.VGG_19:
                net = VGG_19(in_channel=in_channel*timeSequenze, num_classes=num_classes)
            
            case _:
                raise ValueError(f"Invalid network type: {networkType}")
            
    
        match trainingModel:
            case TrainModel_Type.Predictions:
                raise NotImplementedError("Predictions is not implemented yet")

            case TrainModel_Type.ImageClassification:
                trainer = LightTrainerModel_ImageClassification(net=net, lr=trainer_lr)

            case TrainModel_Type.ObjectDetection:
                raise NotImplementedError("ObjectDetection is not implemented yet")

            case TrainModel_Type.SemanticSegmentation:
                raise NotImplementedError("SemanticSegmentation is not implemented yet")

            case _:
                raise ValueError(f"Invalid training model: {trainingModel}")
    
    
        return net, trainer
    
        
    
    
    