from multiprocessing import process
import threading
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from functools import lru_cache
from abc import ABC,abstractmethod
import numpy as np

from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre


# class Segmentation_DatasetBase(Dataset):
    
#     def __init__(self, classesCount: tuple, transform: transforms = None, oneHot: bool = False) -> None:
#         super().__init__()
        
#         self._transform: transforms = transform
#         self._oneHot: bool = oneHot



class DatasetBase(Dataset):

    def __init__(self, classesCount: int, x_transform: transforms = None, y_transform: transforms = None, oneHot: bool = False, device = None, caching:bool = True) -> None:
        super().__init__()

        self._x_transform: transforms = x_transform
        self._y_transform: transforms = y_transform
        self._oneHot: bool = oneHot
        self._classesCount: int = classesCount
        self._device = device
        self._DatasetSize: int = None
        self._caching = caching
    
        self._load_only_Y: bool = False
        self._skip_transforms: bool = False
        
    def setLoadOnlyY(self, load_only_Y: bool):
        self._load_only_Y = load_only_Y
        
    def setSkipTransforms(self, skip_transforms: bool):
        self._skip_transforms = skip_transforms


    def __getitem__(self, idx: int) -> any:
        if self._load_only_Y:
            return torch.zeros(1), self.get_y_value(idx)
        
        itemDict = self._getItem(idx)
        
        if not self._skip_transforms:
            itemDict = self.on_apply_transforms(itemDict)
        
        itemDict = self.adjustData(itemDict)
        
        
        return  itemDict['x'], itemDict['y']
    
    def getItems(self, idx: int) -> Dict[str,any]:
        return self._getItem(idx)
    
    def __len__(self) -> int:
        if self._DatasetSize is None:
            self._DatasetSize = self._getSize()
        return self._DatasetSize
    
    @lru_cache(maxsize=None)  # Decoratore per memorizzare i risultati calcolati
    def _one_hot_encode(self, labels: torch.tensor) -> torch.tensor:
        labels_tensor = labels.clone().detach().long()
        one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=self._classesCount)
        return one_hot

    
    def _one_hot_encode_no_cache(self, labels: any):
        labels_tensor = labels.clone().detach().long()
        one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=self._classesCount)
        return one_hot

    @abstractmethod
    def get_y_value(self, idx: int) -> Optional[torch.Tensor]:
        raise NotImplementedError("get_y_value method must be implemented")

    @abstractmethod
    def _getItem(self, idx: int) -> any:
        pass
    
    @abstractmethod
    def _getY_Item(self, idx: int) -> any:
        pass

    @abstractmethod
    def _getSize(self) -> int:
        pass
    
    @abstractmethod
    def worker_init_fn(self, worker_id: int) -> None:
        pass
    
    @abstractmethod
    def adjustData(self, items: dict[str, any]) -> dict[str, any]:
        return items
    
    @abstractmethod
    def on_apply_transforms(self, items: dict[str, any]) -> dict[str, any]:
        ...
    
    @staticmethod
    def _FormatResult(func):
        def wrapper(self, *args, **kwargs):
            
            data:  np.array | torch.Tensor | None = None
            label: np.array | torch.Tensor |int | None = None
            
            data_Tensor: torch.Tensor = None
            label_Tensor: torch.Tensor = None
            
        
            data, label = func(self, *args, **kwargs)
            
            
            if type(data) == torch.Tensor:
                data = data.numpy()
                
            if type(label) == torch.Tensor:
                label = label.numpy()
            
            if self._y_transform:
                label_Tensor = self._y_transform(label)
                label_Tensor = label_Tensor.long()
            else:
                label_Tensor = torch.Tensor(label)
            
            if self._oneHot:
                if self._caching:
                    label_Tensor = torch.Tensor(self._one_hot_encode(label))
                else:
                    label_Tensor = self._one_hot_encode_no_cache(label)
   
            
            if self._x_transform:
                data_Tensor = self._x_transform(data)
            else:
                data_Tensor = torch.Tensor(data)
                
            
            return data_Tensor, label_Tensor
        return wrapper
    
    

class PostgresDB_Dataset(DatasetBase):
    
    __lock = threading.Lock()
    __processStream: dict[int, PostgresDB] = {}
    

    def __init__(self, connectionParametre: DatabaseParametre, classesCount: int, transform: transforms, oneHot: bool) -> None:
        super().__init__(classesCount, transform, None, oneHot)
        
        assert (connectionParametre is not None) 
        assert (type(connectionParametre) is DatabaseParametre)
        
        self._parametre: DatabaseParametre = connectionParametre
        self._postgresDB = None
        
        if(self._postgresDB is None):
           self.__createStream()
           
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        
        if hasattr(self, '_postgresDB'):
            if(self._postgresDB is None):
                self.__createStream()
     
    def __getstate__(self) -> object:
        
        if hasattr(self, '_postgresDB') and self._postgresDB is not None:
            self._postgresDB.close_pool()
            self._postgresDB = None
        
        state = self.__dict__.copy()
        
        if '_postgresDB' in state:
            del state['_postgresDB']

        return state
     
    def __del__(self) -> None:
        #self.closeAllConnections()
        pass
    
    def worker_init_fn(self, worker_id: int) -> None:
        self._postgresDB = PostgresDB(self._parametre)
        #print(f"Worker {worker_id} DB connection created")
        
        #print(f"Worker {worker_id}: {self._postgresDB}")
        
   
        
    @staticmethod
    def __synchronized(lock):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with lock:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    
    def __createStream(self) -> None:
        """Crea una connessione solo se non è già presente per il processo."""
        
        if not hasattr(self, '_postgresDB'):
            self._postgresDB = PostgresDB(self._parametre)
        elif self._postgresDB is None:
            self._postgresDB = PostgresDB(self._parametre)
    
    # @__synchronized(__lock)
    # def closeAllConnections(self) -> None:
    #     """Chiude la connessione per il processo specificato."""
        
    #     for stream in PostgresDB_Dataset.__processStream.values():
    #         stream.close_pool()
        
    #     PostgresDB_Dataset.__processStream.clear()
            
    def _getStream(self) -> PostgresDB:
        if not hasattr(self, '_postgresDB'):
            self.__createStream()
        
        return self._postgresDB
    
    # @__synchronized(__lock)
    # def _getStream(self) -> PostgresDB:
    #     """Restituisce la connessione per il processo specificato."""
        
    #     processID = processID = process.current_process().pid
        
    #     if not self._isConnected(processID):
    #         self.__createStream(processID)
            
    #     return PostgresDB_Dataset.__processStream[processID] 
        
    #     return self._postgresDB     
        