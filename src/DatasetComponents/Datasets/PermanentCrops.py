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
from typing import Dict
import pickle



class TileName(Enum):
    SPAGNA = "31TCF"
    PUGLIA = "33SVB"
    SICILIA = "33TXF"


class PermanentCrops(Segmentation_Dataset_Base):
    
    class Distance(Enum):
        M10 = "_10m.tif"
        M20 = "_20m.tif"
        M60 = "_60m.tif"
    
    TemporalSize: Final[int] = 62
    ImageChannels: Final[int] = 13
    ImageWidth: Final[int] = 48*2
    ImageHeight: Final[int] = 48*2
    ClassesCount: Final[int] = 4
    
    TRAINIG_E_VALIDATION_TILES: Final[List[str]] = [TileName.SICILIA.value, TileName.PUGLIA.value]
    TEST_TILES: Final[List[str]] = [TileName.SPAGNA.value]
    _DISTANCE_LIST: Final[List[str]] = [Distance.M10.value, Distance.M20.value, Distance.M60.value]
    _MAP_Y_VALUES = lambda y: 0 if y == 0 else y - 220
    
    TRAIN_VAL_SPLIT_PERCENTAGE = 0.8
    _PACHES_COUNT_DICT_KEY = "patches"
    _CACHE_DATA = True
    _CACHE_FOLDER = os.path.join(Globals.TEMP_DATA, f"PermanentCrops_tempData")
    
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
        
        file = os.path.join(PermanentCrops._CACHE_FOLDER, f"{mode}.pickle")
        serialized: bool = False
        totalPaches: int = 0
        
        if PermanentCrops._CACHE_DATA:
            if not os.path.exists(PermanentCrops._CACHE_FOLDER):
                os.makedirs(PermanentCrops._CACHE_FOLDER)
        
        if PermanentCrops._CACHE_DATA and os.path.exists(file):
            Globals.APP_LOGGER.info(f"Loading cached data...")
            with open(file, "rb") as f:
                self._dataDict = pickle.load(f)
            serialized = True
        
        if not serialized:
            match mode:
                case DatasetMode.TRAINING:    
                    Globals.APP_LOGGER.info(f"Loading TRAINING pacheses...")                
                    
                    for tile in PermanentCrops.TRAINIG_E_VALIDATION_TILES:
                        Globals.APP_LOGGER.info(f"processing: {tile}")
                        
                        tilePath = os.path.join(self._folderPath, tile)
                        folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                        folders = np.array(folders, dtype=object)

                        split_index = int(len(folders) * PermanentCrops.TRAIN_VAL_SPLIT_PERCENTAGE)
                    
                        training =  folders[:split_index]
                        
                        self._dataDict[tile] = {
                            "data" : training,
                            "range" : (totalPaches, totalPaches + len(training))
                        }
                        totalPaches += len(training)
                        
                    self._dataDict[PermanentCrops._PACHES_COUNT_DICT_KEY] = totalPaches

                case DatasetMode.VALIDATION:
                    Globals.APP_LOGGER.info(f"Loading VALIDATION pacheses...")              
                    for tile in PermanentCrops.TRAINIG_E_VALIDATION_TILES:
                        Globals.APP_LOGGER.info(f"processing: {tile}")
                        
                        tilePath = os.path.join(self._folderPath, tile)
                        folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                        split_index = int(len(folders) * PermanentCrops.TRAIN_VAL_SPLIT_PERCENTAGE)

                        validation =  folders[split_index:]

                        self._dataDict[tile] = {
                            "data" : validation,
                            "range" : (totalPaches, totalPaches + len(validation))
                        }
                        totalPaches += len(validation)

                    self._dataDict[PermanentCrops._PACHES_COUNT_DICT_KEY] = totalPaches
                    
                case DatasetMode.TEST:

                    Globals.APP_LOGGER.info(f"Loading TEST pacheses...")
                    for tile in PermanentCrops.TEST_TILES:
                        Globals.APP_LOGGER.info(f"processing: {tile}")
                        
                        tilePath = os.path.join(self._folderPath, tile)
                        folders = sorted(f for f in os.listdir(tilePath) if os.path.isdir(os.path.join(tilePath, f)))

                        self._dataDict[tile] = {
                            "data" : folders,
                            "range" : (totalPaches, totalPaches + len(folders))
                        }
                        totalPaches += len(folders)

                    self._dataDict[PermanentCrops._PACHES_COUNT_DICT_KEY] = totalPaches
                    
                case _:
                    raise Exception(f"Invalid mode {mode}") 
         
        if PermanentCrops._CACHE_DATA and not serialized:
            with open(file, "wb") as f:
                pickle.dump(self._dataDict, f)
            serialized = True
            
        Globals.APP_LOGGER.info(f"{mode} PermanentCrops dataset total paches: {totalPaches}")
        for key in self._dataDict.keys():
            if key != PermanentCrops._PACHES_COUNT_DICT_KEY:
                Globals.APP_LOGGER.info(f"{mode} PermanentCrops dataset {key} paches range: {self._dataDict[key]['range']}")
                
                
                
                
    def _mapIndex(self, index: int) -> str:
        '''
        Dall'indice ritorna il percorso del folder
        '''
        
        for keys in self._dataDict.keys():
            if keys == PermanentCrops._PACHES_COUNT_DICT_KEY:
                continue

            patchData: Dict[str, any] = self._dataDict[keys]
            idx_min = patchData["range"][0]
            idx_max = patchData["range"][1]

            if index >= idx_min and idx_max:
                idx = index - idx_min
                data_folder_path = os.path.join(self._folderPath, keys,  patchData["data"][idx])

                return data_folder_path
                
        raise Exception(f"Index {index} not found in any year range")   
    
    def get_dates(self, path: str, sample_number= int | None) -> list[str]:
        
        '''
        Ottengo i nomi dei file della sequenza temporale
        '''
        
        #assert os.path.exists(path), f"Path {path} does not exist"
    
        files = os.listdir(path)
        dates = list()
        
        for f in files:
            date = f.split("_")[0]
            if len(date) == 8:  # 20160101
                dates.append(date)

        dates = list(set(dates))
        
        if sample_number is None or sample_number < 0:
            return dates
        
        if len(dates) > sample_number:
            dates = random.sample(dates, sample_number)
        
        elif len(dates) < sample_number:
            while len(dates) < sample_number:
                dates.append(random.choice(dates))
        
        dates.sort()
        return dates
    
    def _normalize_dif_data(self, data: np.array, profile) -> np.array:
        # return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        
        match(profile['dtype']):
            case "uint8":
                return data / 255.0

            case "uint16":
                return data / 65535.0
            
            case "uint32":
                return data / 4294967295.0
        
            case _ :
                raise Exception(f"Invalid data type {profile['dtype']}")
            
    
    def _load_dif_file(self, filePath:str, normalize: bool = True) -> Dict[str,any]:
        
        print(f"Loading {filePath}")
        
        with rasterio.open(filePath) as src:
            data = src.read()
            data = data.astype(np.float32)
            profile = src.profile
            
        if normalize:
            data = self._normalize_dif_data(data = data, profile = profile)
        return {"data" : data, "profile" : profile}
    
    def _load_y(self, folder:str) -> np.array:
        y = self._load_dif_file(filePath=os.path.join(folder, "y.tif"), normalize = False)["data"]
        y = y.astype(np.uint8)
        y = np.where(y == 0, 220, y)
        y = y - 220
        
        return y
    
    def _load_year_sequenze(self, idx: str) -> dict[str, any]:
        
        sequenzeFolder:str = self._mapIndex(idx)
        dates: List[str] = self.get_dates(path=sequenzeFolder, sample_number=PermanentCrops.TemporalSize)
        profile = None
        
        x: torch.Tensor = torch.empty((PermanentCrops.TemporalSize, PermanentCrops.ImageChannels, PermanentCrops.ImageHeight, PermanentCrops.ImageWidth), dtype=torch.float32)

        # Itera attraverso ogni data per caricare i dati temporali
        for t, date in enumerate(dates):
            current_channel_index = 0  # Reset per ogni t-step

            # Carica i dati per ciascuna distanza selezionata
            for suffix in PermanentCrops._DISTANCE_LIST:
                DataDict = self._load_dif_file(os.path.join(sequenzeFolder, f"{date}{suffix}"))
                data = DataDict['data']
                
                if profile is None:
                    profile = DataDict["profile"]
                
                # Aggiungi dimensione per l'interpolazione
                tensor = torch.from_numpy(data).unsqueeze(0)  
                
                if suffix != PermanentCrops.Distance.M10.value:
                    tensor = F.interpolate(tensor, size=(PermanentCrops.ImageHeight, PermanentCrops.ImageWidth))
                
                num_channels = tensor.size(1)
                
                # Assegna il tensor ai canali specifici per il timestamp t
                x[t, current_channel_index:current_channel_index + num_channels, :, :] = tensor.squeeze(0)
                current_channel_index += num_channels  # Aggiorna l'indice dei canali

        # Carica la maschera di segmentazione
        y = self._load_y(sequenzeFolder)
        return {"x": x, "y": y, "profile": profile}
    
    
    
    def _getSize(self) -> int:
        return self._dataDict[PermanentCrops._PACHES_COUNT_DICT_KEY]   
    
    
    def _getItem(self, idx: int) -> dict[str, any]:
        assert idx >= 0 and idx < self.__len__(), f"Index {idx} out of range"
        
        dictData: Dict[str, any] = self._load_year_sequenze(idx)
    
        
        x = dictData["x"]
        y = dictData["y"]
                
        if not self._useTemporalSize:
            x = x.view(-1, PermanentCrops.ImageHeight, PermanentCrops.ImageWidth)
            #x = x.permute(1, 2, 0)
            # x = np.transpose(x, (1, 2, 0))
        else:
            # permute channels with time_series (t x c x h x w) -> (c x t x h x w)
            x = x.permute(1, 0, 2, 3)
        
        #y = torch.squeeze(y, dim=0)
        #(1, 48, 48) -> (48, 48)
        y = np.squeeze(y, axis=0)
        y = torch.from_numpy(y)
        
        #x = x.float()
        
        
        y = torch.Tensor(self._one_hot_encode_no_cache(y))
        #y = y.float()
        y = y.permute(2,0,1)# -> (c x h x w)
   
        dictData['x'] = x
        dictData['y'] = y
        
        dictData['info'] = dictData['profile']
        dictData.pop('profile', None)
        
        #print(x.shape, y.shape) ---> torch.Size([48, 48, 832]) np.size(48, 48)
    
        return dictData   
    
    
    def on_apply_transforms(self, items: dict[str, any]) -> dict[str, any]:
        
        x = items['x']
        y = items['y']
        
        if self._x_transform is not None:
            
            #y = y.unsqueeze(0)
            
            if self._useTemporalSize:
        
                #torch.Size([27, 48, 48]) -> torch.Size([1, 27, 48, 48])
                y = y.unsqueeze(0)
                
                #torch.Size([1, 27, 48, 48]) -> torch.Size([13, 27, 48, 48])
                y = y.repeat(13, 1, 1, 1)
                
                
                #torch.Size([13, 32, 48, 48]) ° torch.Size([13, 27, 48, 48])
                # ->torch.Size([13, 59, 48, 48])
                xy = torch.cat((x, y), dim=1)
                xy = self._x_transform(xy)
                
                # Separare x e y dopo la trasformazione
                x = xy[:, :PermanentCrops.TemporalSize, :, :]  # I primi 32 canali vanno a x
                y = xy[:, PermanentCrops.TemporalSize:, :, :]  # I successivi 27 canali vanno a y
                
                #torch.Size([13, 27, 48, 48]) -> torch.Size([27, 48, 48])
                #y = y[0, 0, :, :]
                y = y[0, :, :, :]
            else:


                xy = torch.cat((x, y), dim=0)
                xy = self._x_transform(xy)
                x = xy[:PermanentCrops.TemporalSize*PermanentCrops.ImageChannels, :, :]
                y = xy[PermanentCrops.TemporalSize*PermanentCrops.ImageChannels:, :, :]
                #y = y[0, :, :]

        
        # if self._oneHot:
        #     y = torch.Tensor(self._one_hot_encode_no_cache(y))
        
        # y = y.float()
        # y = y.permute(2,0,1)# -> (c x h x w)

        items['x'] = x
        items['y'] = y
        
        
        return items  
    
    def adjustData(self, items: dict[str, any]) -> dict[str, any]:
        # y = items['y']
        # items['y'] = y.long()
        return items