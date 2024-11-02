from typing import Final
import pytorch_lightning as pl
import torch
import opendatasets

from DatasetComponents.Datasets.munich480 import *

from .DataModuleBase import *
from torch.utils.data import DataLoader
#from DatasetComponents.Datasets.munich480 import *


class Munich480_DataModule(DataModuleBase):
    
    _KAGGLE_DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480"
    
    def __init__(self, datasetFolder:str, download: bool = False, batch_size: int = 1, num_workers: int  = 1):
        super().__init__(datasetFolder, batch_size, num_workers)

        self._download = download
        self._TRAIN: Munich480 | None = None
        self._VAL: Munich480 | None = None
        self._TEST: Munich480 | None = None
        
        
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
        

    def prepare_data(self) -> None:
        if self._download:
            self._DownloadDataset(url= Munich480_DataModule._KAGGLE_DATASET_URL, folder= self._datasetFolder)
    

    def setup(self, stage=None) -> None:
        self._TRAIN = Munich480(self._datasetFolder, mode= Munich480.DataType.TRAINING, year= Munich480.Year.Y2016, transforms=self._training_trasforms)
        self._VAL   = Munich480(self._datasetFolder, mode= Munich480.DataType.VALIDATION, year= Munich480.Year.Y2016, transforms=self._test_trasforms)
        self._TEST  = Munich480(self._datasetFolder, mode= Munich480.DataType.TEST, year= Munich480.Year.Y2016, transforms=self._test_trasforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=True, persistent_workers=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=True, persistent_workers=True, drop_last=True)