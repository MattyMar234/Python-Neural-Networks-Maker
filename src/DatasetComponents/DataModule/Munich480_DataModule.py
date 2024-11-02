from ast import Tuple
from typing import Final
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
import torch
import opendatasets

from DatasetComponents.Datasets.munich480 import Munich480

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os
import os
import os
#from DatasetComponents.Datasets.munich480 import *


class Munich480_DataModule(DataModuleBase):
    
    TemporalSize: Final[int] = 32
    ImageChannels: Final[np.array] = np.array([4,6,3])
    ImageWidth: Final[int] = 48
    ImageHeight: Final[int] = 48
    
    ClassesCount: Final[int] = 27
    
    _KAGGLE_DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480"
    
    def __init__(
        self, 
        datasetFolder:str, 
        distance: Munich480.Distance = Munich480.Distance.m10, 
        year: Munich480.Year = Munich480.Year.Y2016, 
        download: bool = False, 
        batch_size: int = 1, 
        num_workers: int  = 1,
        useTemporalSize: bool = False
    ):
        super().__init__(datasetFolder, batch_size, num_workers)

        assert os.path.exists(datasetFolder), f"La cartella {datasetFolder} non esiste"
        assert type(distance) == Munich480.Distance, f"distance deve essere di tipo Munich480.Distance"
        assert type(year) == Munich480.Year, f"year deve essere di tipo Munich480.Year"
        
        
        self._download = download
        self._TRAIN: Munich480 | None = None
        self._VAL: Munich480 | None = None
        self._TEST: Munich480 | None = None
        
        self._distance = distance
        self._year = year
        self._persistent_workers = True
        self._pin_memory = True
        self._useTemporalSize = useTemporalSize
        self._total_channel = 0
        
        if Munich480.Distance.m10 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[0]
        if Munich480.Distance.m20 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[1]
        if Munich480.Distance.m60 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[2]
            
        if not useTemporalSize:
            self._total_channel *= Munich480_DataModule.TemporalSize * (len(year))
        
        
        self._training_trasforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
            #transforms.Resize((572, 572)),
            transforms.ToTensor() 
        ])
        
        self._test_trasforms = transforms.Compose([
            #transforms.Resize((572, 572)),
            transforms.ToTensor()
        ])
        
    def number_of_channels(self) -> int:
        return self._total_channel
    
    def number_of_classes(self) -> int:
        return Munich480_DataModule.ClassesCount
        
    def input_size(self) -> list[int]:
        if self._useTemporalSize:
            return [1, Munich480_DataModule.TemporalSize, self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]
        else: 
            return [1,self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]

    def prepare_data(self) -> None:
        if self._download:
            self._DownloadDataset(url= Munich480_DataModule._KAGGLE_DATASET_URL, folder= self._datasetFolder)
    

    def setup(self, stage=None) -> None:
        self._TRAIN = Munich480(self._datasetFolder, mode= Munich480.DataType.TRAINING, year= self._year, distance=self._distance, transforms=self._training_trasforms)
        self._VAL   = Munich480(self._datasetFolder, mode= Munich480.DataType.VALIDATION, year= self._year, distance=self._distance, transforms=self._test_trasforms)
        self._TEST  = Munich480(self._datasetFolder, mode= Munich480.DataType.TEST, year= self._year, distance=self._distance, transforms=self._test_trasforms)



    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True)