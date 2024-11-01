from typing import Final
import pytorch_lightning as pl
import torch
import opendatasets

from .DataModuleBase import *
#from DatasetComponents.Datasets.munich480 import *
from src.DatasetComponents.Datasets.munich480 import Munich480

class Munich480_DataModule(DataModuleBase):
    
    _KAGGLE_DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480"
    
    def __init__(self, datasetFolder:str, download: bool = False, batch_size: int = 1, num_workers: int  = 1):
        super().__init__(datasetFolder, batch_size, num_workers)

        t: Munich480 = Munich480()

    def prepare_data(self) -> None:
        if self._download:
            self._DownloadDataset(url= Munich480_DataModule._KAGGLE_DATASET_URL, folder= self._datasetFolder)
    

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass