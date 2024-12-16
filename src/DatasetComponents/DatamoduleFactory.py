from argparse import Namespace
from enum import Enum

from DatasetComponents.DataModule.PermanentCrops_DataModule import PermanentCrops_DataModule
from DatasetComponents.Datasets.DatasetBase import DatasetBase
from DatasetComponents.DataModule.Munich480_DataModule import *


class AvailableDatabodule(Enum):
    
    MNIST = "MNIST"
    CIFAR10 = "Cifar10"
    CIFAR100 = "Cifar100"
    IMAGENET = "Imagenet"
    MUNICH_2D = "Munich_2D"
    MUNICH_3D = "Munich_3D"
    MUNICH_2D_postgres = "Munich_2D_postgres"
    MUNICH_3D_postgres = "Munich_3D_postgres"
    PERMANENT_CROPS = "PermanentCrops"
    
    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())
    
    
def makeDatamodule(datasetName: str, args: Namespace) -> DatasetBase:
    
    value: AvailableDatabodule = AvailableDatabodule(datasetName)
    
    match value:
        
        case AvailableDatabodule.MUNICH_2D:
            return Munich480_DataModule(
                datasetFolder = os.path.join(Globals.DATASET_FOLDER, "munich480"),
                batch_size=args.batch_size,
                num_workers=args.workers,
                useTemporalSize=False,
                year= Munich480.Year.Y2016,
                args = args
            ) 
            
        case AvailableDatabodule.MUNICH_3D:
            return Munich480_DataModule(
                datasetFolder = os.path.join(Globals.DATASET_FOLDER, "munich480"),
                batch_size=args.batch_size,
                num_workers=args.workers,
                useTemporalSize=True,
                year= Munich480.Year.Y2016,
                args = args
            )
            
        case AvailableDatabodule.PERMANENT_CROPS:
            return PermanentCrops_DataModule(
                datasetFolder = os.path.join(Globals.DATASET_FOLDER, "permanent_crops", "dataset96"),
                batch_size=args.batch_size,
                num_workers=args.workers,
                useTemporalSize = True,
                args = args
            )
            
        case _:
            raise Exception(f"Invalid dataset name {value}")