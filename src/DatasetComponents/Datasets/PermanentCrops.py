from argparse import Namespace
import math
import random
from matplotlib import pyplot as plt
import matplotlib
import rasterio

from DatasetComponents.DataModule.DataModuleBase import DatasetMode
import Globals


from .ImageDataset import *
from .DatasetBase import *
from PIL import Image

import torch
import torch.nn.functional as F

import os
from os import listdir
from enum import Enum, auto, Flag
from typing import Final, List, Tuple
import asyncio
import aiofiles

from threading import Lock
from functools import lru_cache



class TileName(Enum):
    SPAGNA = "31TCF"
    PUGLIA = "33SVB"
    SICILIA = "33TXF"


class PermanentCrops(Segmentation_Dataset_Base):
    TemporalSize: Final[int] = 167
    ImageChannels: Final[int] = 13
    ImageWidth: Final[int] = 48*2
    ImageHeight: Final[int] = 48*2
    ClassesCount: Final[int] = 4
    
    TRAINIG_E_VALIDATION_TILES = [TileName.SICILIA, TileName.PUGLIA]
    TEST_TILES = [TileName.SPAGNA]
    TRAIN_VAL_SPLIT_PERCENTAGE = 0.8
    _PACHES_DICT_KEY = "patches"
    
    
    def __setstate__(self, state):
        return
     
    def __getstate__(self) -> object:
        return {} 
    
    
    def __init__(self, folderPath:str | None, mode: DatasetMode, transforms, useTemporalSize: bool = False, args: Namespace | None = None):
        
        assert type(mode) == DatasetMode, "Invalid mode type"
        
        
        self._folderPath:str | None = folderPath
        self._dataSequenze: np.array | None = None
        self._useTemporalSize: bool = useTemporalSize
        self._mode = mode
        self._dataDict: Dict[str, any] | None = {}
        
        Segmentation_Dataset_Base.__init__(
            self, 
            imageSize = (PermanentCrops.ImageWidth, PermanentCrops.ImageHeight, PermanentCrops.ImageChannels, PermanentCrops.TemporalSize if not useTemporalSize else 0), 
            classesCount = PermanentCrops.ClassesCount, 
            x_transform=transforms,
            y_transform = transforms,
            oneHot = True,
            args = args
        )
        
        totalPaches: int = 0
        
        match mode:
            case DatasetMode.TRAINING:
                Globals.APP_LOGGER.info(f"Loading TRAINING pacheses...")                
                for tile in PermanentCrops.TRAINIG_E_VALIDATION_TILES:
                    Globals.APP_LOGGER.info(f"processing: {tile.value}")
                    
                    tilePath = os.path.join(self._folderPath, tile.value)
                    folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                    folders = np.array(folders, dtype=object)

                    split_index = int(len(folders) * PermanentCrops.TRAIN_VAL_SPLIT_PERCENTAGE)
                
                    training =  folders[:split_index]
                    
                    self._dataDict[tile] = {
                        "data" : training,
                        "range" : (totalPaches, totalPaches + len(training))
                    }
                    totalPaches += len(training)
                    
                self._dataDict[PermanentCrops._PACHES_DICT_KEY] = totalPaches
            
            
            case DatasetMode.VALIDATION:
                Globals.APP_LOGGER.info(f"Loading VALIDATION pacheses...")              
                for tile in PermanentCrops.TRAINIG_E_VALIDATION_TILES:
                    Globals.APP_LOGGER.info(f"processing: {tile.value}")
                    
                    tilePath = os.path.join(self._folderPath, tile.value)
                    folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                    split_index = int(len(folders) * PermanentCrops.TRAIN_VAL_SPLIT_PERCENTAGE)

                    validation =  folders[split_index:]

                    self._dataDict[tile] = {
                        "data" : validation,
                        "range" : (totalPaches, totalPaches + len(validation))
                    }
                    totalPaches += len(validation)

                self._dataDict[PermanentCrops._PACHES_DICT_KEY] = totalPaches
                
            case DatasetMode.TEST:

                Globals.APP_LOGGER.info(f"Loading TEST pacheses...")
                for tile in PermanentCrops.TEST_TILES:
                    Globals.APP_LOGGER.info(f"processing: {tile.value}")
                    
                    tilePath = os.path.join(self._folderPath, tile.value)
                    folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                    self._dataDict[tile] = {
                        "data" : folders,
                        "range" : (totalPaches, totalPaches + len(folders))
                    }
                    totalPaches += len(folders)

                self._dataDict[PermanentCrops._PACHES_DICT_KEY] = totalPaches
                
          
            case _:
                raise Exception(f"Invalid mode {mode}") 
            
        Globals.APP_LOGGER.info(f"{mode} PermanentCrops dataset total paches: {totalPaches}")
        for key in self._dataDict.keys():
            if key != PermanentCrops._PACHES_DICT_KEY:
                Globals.APP_LOGGER.info(f"{mode} PermanentCrops dataset {key} paches range: {self._dataDict[key]['range']}")