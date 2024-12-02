import pickle
from typing import Dict, Final, List
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
from psycopg2 import Binary
import pytorch_lightning as pl
from torchvision import transforms
import torch
import opendatasets
import colorsys
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import TableBase, TensorTable
from DatasetComponents.Datasets.DatasetBase import PostgresDataset_Interface
from DatasetComponents.Datasets.PermanentCrops import PermanentCrops
from DatasetComponents.Datasets.munich480 import Munich480
import Globals
from Networks.Metrics.ConfusionMatrix import ConfusionMatrix
from Networks.NetworkComponents.NeuralNetworkBase import *
from Utility.TIF_creator import TIF_Creator

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os
import time



class PermanentCrops_DataModule(DataModuleBase):
    
    TemporalSize: Final[int] = 167
    ImageChannels: Final[int] = 13
    ImageWidth: Final[int] = 48*2
    ImageHeight: Final[int] = 48*2
    ClassesCount: Final[int] = 4
    
    _SINGLETON_INSTANCE = None
    _ONE_HOT: Final[bool] = True
    

    
    def __setstate__(self, state):
        return
     
    def __getstate__(self) -> object:
        return {}
    
    #singleton
    def __new__(cls, *args, **kwargs):
        if cls._SINGLETON_INSTANCE is None:
            cls._SINGLETON_INSTANCE = super().__new__(cls)
        return cls._SINGLETON_INSTANCE
    
    def __init__(
        self, 
        datasetFolder:str, 
        batch_size: int = 1, 
        num_workers: int  = 1,
        useTemporalSize: bool = True,
        args: Namespace | None = None
    ):
        
        super().__init__(datasetFolder, batch_size, num_workers, args)
        
        
        self._TRAINING: PermanentCrops | None = None
        self._VALIDATION: PermanentCrops | None = None
        self._TEST: PermanentCrops | None = None
        self._setup_done = False
        self._useTemporalSize = useTemporalSize
        
        self._training_trasforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            
        ])
        
        self._test_trasforms = transforms.Compose([
            
        ])
        
    @property 
    #@lru_cache
    def use_oneHot_encoding(self) -> bool:
        return PermanentCrops_DataModule._ONE_HOT
    
    # @property 
    # def getIgnoreIndexFromLoss(self) -> int:
    #     return 0
    
    @property
    def input_channels(self) -> int:
        return PermanentCrops_DataModule.ImageChannels
    
    @property
    def output_classes(self) -> int:
        return PermanentCrops_DataModule.ClassesCount
    
    @property    
    def input_size(self) -> list[int]:
        if self._useTemporalSize:
            return [1, PermanentCrops_DataModule.ImageChannels, PermanentCrops_DataModule.TemporalSize, PermanentCrops_DataModule.ImageHeight, PermanentCrops_DataModule.ImageWidth]
        else: 
            return [1, PermanentCrops_DataModule.ImageChannels, PermanentCrops_DataModule.ImageHeight, PermanentCrops_DataModule.ImageWidth]
        
    def setup(self, stage=None) -> None:
        if self._setup_done:
            return
        
        self._TRAINING   = PermanentCrops(args=self._args, folderPath = self._datasetFolder, mode= DatasetMode.TRAINING, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        self._VALIDATION = PermanentCrops(args=self._args, folderPath = self._datasetFolder, mode= DatasetMode.VALIDATION, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._TEST       = PermanentCrops(args=self._args, folderPath = self._datasetFolder, mode= DatasetMode.TEST, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._setup_done = True
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAINING, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        #return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VALIDATION, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def calculate_classes_weight(self) ->torch.tensor:
        if not self._setup_done:
            self.setup()
            
            
    def on_work(self, model: ModelBase, device: torch.device, **kwargs) -> None:
        print(self._TRAINING[0])