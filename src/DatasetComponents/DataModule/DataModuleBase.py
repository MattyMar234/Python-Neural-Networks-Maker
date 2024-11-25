from argparse import Namespace
from ast import Tuple
from typing import List, Optional
import numpy as np
import opendatasets
import pytorch_lightning as pl
import torch

from abc import abstractmethod
import os

#from Networks.NetworkComponents.NeuralNetworkBase import ModelBase


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, datasetFolder:str, batch_size: int = 1, num_workers: int  = 1, args: Namespace | None = None):
        super().__init__()
        
        assert batch_size > 0, "Batch size must be greater than 0"
        assert num_workers >= 0, "Number of workers must be greater than or equal to 0"
        assert isinstance(datasetFolder, str), "Dataset folder must be a string"
        assert len(datasetFolder) > 0, "Dataset folder must not be empty"

        self._datasetFolder = datasetFolder
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._classes_weights: torch.Tensor | None = None
        self._args: Namespace | None = args

        
    def _DownloadDataset(self, url:str, folder:str) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        opendatasets.download(dataset_id_or_url=url, data_dir=folder)
    
    @abstractmethod 
    def map_classes(self, classes: np.ndarray | List[int] | int) -> List[str] | str | None:
        #raise NotImplementedError("map_classes method must be implemented")
        return None
      
    @abstractmethod
    def classesToIgnore(self) -> List[int]: 
        raise NotImplementedError("classesToIgnore method must be implemented")
    

    @abstractmethod
    def calculate_classes_weight(self) -> List[int]:
        return torch.ones(self.output_classes) * (1 / self.output_classes)
        #raise NotImplementedError("classes_frequenze method must be implemented")
    
    @property 
    def getWeights(self) -> torch.Tensor:
        if self._classes_weights is None:
            self._classes_weights = self.calculate_classes_weight()
        
        return self._classes_weights
    
    @property 
    @abstractmethod
    def getIgnoreIndexFromLoss(self) -> int:
        return -100
    
    @property 
    @abstractmethod
    def use_oneHot_encoding(self) -> bool:
        raise NotImplementedError("use_oneHot_encoding method must be implemented")
    
    @property 
    @abstractmethod  
    def output_classes(self) -> int:
        ...
    
    @property
    @abstractmethod 
    def input_channels(self) -> int:
        ...
        
    @property  
    @abstractmethod  
    def input_size(self) -> list[int] | None:
        return None
    

    @abstractmethod
    def prepare_data(self) -> None:
        ...

    @abstractmethod
    def setup(self, stage=None):
        ...

    @abstractmethod
    def train_dataloader(self):
        ...

    @abstractmethod
    def val_dataloader(self):
        ...
        
    @abstractmethod
    def test_dataloader(self):
        ...
    
    @abstractmethod
    def show_processed_sample(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        ...
    
    @abstractmethod 
    def on_work(self, model, device: torch.device,**kwargs) -> None:
        raise NotImplementedError("on_work method must be implemented")

    