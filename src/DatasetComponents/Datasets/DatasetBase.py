from multiprocessing import process
import threading
from typing import Dict
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

    def __init__(self, classesCount: int, transform: transforms = None, oneHot: bool = False, device = None, caching:bool = True) -> None:
        super().__init__()

        self._transform: transforms = transform
        self._oneHot: bool = oneHot
        self._classesCount: int = classesCount
        self._device = device
        self._DatasetSize: int = None
        self._caching = caching


    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._getItem(idx)
    
    def __len__(self) -> int:
        if self._DatasetSize is None:
            self._DatasetSize = self._getSize()
        return self._DatasetSize
    
    @lru_cache(maxsize=None)  # Decoratore per memorizzare i risultati calcolati
    def _one_hot_encode(self, n):
        if n > self._classesCount:
            raise IndexError("Index out of range")
            
        one_hot= np.zeros(self._classesCount, dtype=np.float32)
        one_hot[int(n)] = 1.0
        return one_hot
    
    def _one_hot_encode_no_cache(self, labels_np):
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=self._classesCount)
        return one_hot

    @abstractmethod
    def _getItem(self, idx: int) -> any:
        pass

    @abstractmethod
    def _getSize(self) -> int:
        pass
    
    @abstractmethod
    def worker_init_fn(self, worker_id: int) -> None:
        pass
    
    @staticmethod
    def _FormatResult(func):
        def wrapper(self, *args, **kwargs):
            
            data: any = None
            label:int = None
            
            data_Tensor: torch.Tensor = None
            label_Tensor: torch.Tensor = None
            
            
            data, label = func(self, *args, **kwargs)
            
            if self._oneHot:
                if self._caching:
                    label_Tensor = torch.Tensor(self._one_hot_encode(label))
                else:
                    label_Tensor = self._one_hot_encode_no_cache(label)
                #label_Tensor = torch.Tensor(self._one_hot_encode(label))
            else:
                label_Tensor = torch.LongTensor(label)
                
            
            if self._transform:
                data_Tensor = self._transform(data)
        
                if self._device is not None:
                    data_Tensor = data_Tensor.to(self._device)
            else:
                
                if self._device is not None:
                    data_Tensor = torch.Tensor(data, device=self._device)
                else:
                    data_Tensor = torch.Tensor(data)
            
            return data_Tensor, label_Tensor
        return wrapper
    
    

class PostgresDB_Dataset(DatasetBase):
    
    __lock = threading.Lock()
    __processStream: dict[int, PostgresDB] = {}
    

    def __init__(self, connectionParametre: DatabaseParametre, classesCount: int, transform: transforms, oneHot: bool) -> None:
        super().__init__(classesCount, transform, oneHot)
        
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
        