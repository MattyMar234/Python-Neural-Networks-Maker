
from argparse import Namespace
from enum import Enum, auto
from multiprocessing import process
import os
import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple
from psycopg2 import Binary
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from functools import lru_cache
from abc import ABC,abstractmethod
import numpy as np

from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import TableBase, TensorTable
import Globals
from utility import measure_execution_time


# class Segmentation_DatasetBase(Dataset):
    
#     def __init__(self, classesCount: tuple, transform: transforms = None, oneHot: bool = False) -> None:
#         super().__init__()
        
#         self._transform: transforms = transform
#         self._oneHot: bool = oneHot



class DatasetBase(Dataset):


    def __setstate__(self, state) -> None:
        #imposto i valori
        self.__dict__.update(state)
        
        #if hasattr(self, '_postgresDataset'):
        if self._postgresDB is None and self._useDB:
            self._createStream()
   
     
    def __getstate__(self) -> dict[str, Any]:
        if hasattr(self, '_postgresDB') and self._postgresDB is not None:
            #self._postgresDB.close_pool()
            self._postgresDB = None
        
        state = self.__dict__.copy()
        
        if '_postgresDB' in state:
            state['_postgresDB'] = None

        return state


    def __init__(self, classesCount: int, x_transform: transforms = None, y_transform: transforms = None, oneHot: bool = False, device = None, caching:bool = True, args: Namespace | None = None) -> None:
        super().__init__()

        self._x_transform: transforms = x_transform
        self._y_transform: transforms = y_transform
        self._oneHot: bool = oneHot
        self._classesCount: int = classesCount
        self._device = device
        self._DatasetSize: Optional[int]  = None
        self._caching = caching
    
        self._load_only_Y: bool = False
        self._skip_transforms: bool = False
        
        self._argsDict: Optional[Dict[str, any]] = vars(args) if args is not None else None
        #self._databaseParametre:  Optional[DatabaseParametre] = None
        self._postgresDatasets: Dict[int, PostgresDB] = {}
        self._useDB: bool = False
        self._table: Optional[TensorTable] = None
        
        
        if self._argsDict is not None and self._argsDict[Globals.ENABLE_DATABASE]:
            self._useDB = True
        
        
        
        
        self._createStream()
            
    def __len__(self) -> int:
        if self._DatasetSize is None:
            self._DatasetSize = self._getSize()
        return self._DatasetSize    
       
        
    def __getitem__(self, idx: int) -> any:
        if self._load_only_Y:
            return torch.zeros(1), self.get_y_value(idx)
        
        itemDict = self.getItem(idx)
        
        if not self._skip_transforms:
            itemDict = self.on_apply_transforms(itemDict)
        
        itemDict = self.adjustData(itemDict)
        return  itemDict['x'], itemDict['y']    


    #@lru_cache(maxsize=os.cpu_count() + 4)
    def getStream(self, PID) -> PostgresDB:
        
        PID = 0
        stram = self._postgresDatasets.get(PID, None)
        
        if stram is None:
            print(f"New connection for: {PID}")

            self._postgresDatasets[PID] = self._createStream()
            return self._postgresDatasets[PID]
        
        return stram
        return self._createStream()

    def setLoadOnlyY(self, load_only_Y: bool):
        self._load_only_Y = load_only_Y
        
    def setSkipTransforms(self, skip_transforms: bool):
        self._skip_transforms = skip_transforms


    @abstractmethod
    def _generateTableName(self):
        return self.__class__.__name__.lower()

    def _createStream(self) -> PostgresDB:
        
        if not self._useDB:
            return 
            
        databaseParametre = DatabaseParametre(
            host=self._argsDict[Globals.DB_HOST],#"host.docker.internal",
            port=self._argsDict[Globals.DB_PORT],
            database=self._argsDict[Globals.DB_NAME],
            user=self._argsDict[Globals.DB_USER],
            password=self._argsDict[Globals.DB_PASSWORD],
            maxconn  = Globals.DEFAULT_MAX_CONNECTIONS,
            timeout  = Globals.CONNECTION_TIMEOUT
        )
        
        #print(databaseParametre)
        
        self._table = TensorTable(self._generateTableName())
        database = PostgresDB(databaseParametre)
        database.execute_query(self._table.createTable_Query())
        return database
    
    #@measure_execution_time
    def getItem(self, idx: int) -> Dict[str,any]:
        if not self._useDB:
            return self._getItem(idx)
        
        database = self.getStream(os.getpid())
        query: str = self._table.getElementAt_Query(idx)
        result = database.fetch_results(query)
    
        
        if result is not None:
            if len(result) == 1:
                row = result[0]
                
                return {
                    'x' : pickle.loads(row[1]),
                    'y' : pickle.loads(row[2]),
                    'info' : pickle.loads(row[3])
                }
            
            if len(result) > 1:
                raise Exception("Too many results")
    
        
        data: Dict[str, any] = self._getItem(idx)
        
        query = self._table.insertElement_Query(
            id = idx, 
            x = Binary(pickle.dumps(data['x']))     , 
            y = Binary(pickle.dumps(data['y'])), 
            info = Binary(pickle.dumps(data['info']))
        )
        
        database.execute_query(query)
        return data
    
    
    @abstractmethod
    def getItemInfo(self, idx: int) -> Dict[str,any]:
        raise NotImplementedError("getItemInfo method must be implemented")
    
    
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
    
    

class PostgresDataset_Interface(object):
    
    __lock = threading.Lock()
    __processStream: dict[int, PostgresDB] = {}
    

    def __init__(self, connectionParametre: DatabaseParametre) -> None:
        super().__init__()
        
        assert (connectionParametre is not None) 
        assert (type(connectionParametre) is DatabaseParametre)
        
        self._parametre: DatabaseParametre = connectionParametre
        self._postgresDataset: Optional[PostgresDB] = None
        
        if(self._postgresDataset is None):
           self.__createStream()
           
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        if hasattr(self, '_postgresDataset'):
            if(self._postgresDataset is None):
                self.__createStream()
        else:
            self.__createStream()
     
     
    def __getstate__(self) -> object:
        if hasattr(self, '_postgresDataset') and self._postgresDataset is not None:
            #self._postgresDataset.close_pool()
            self._postgresDataset = None
        
        state = self.__dict__.copy()
        
        if '_postgresDataset' in state:
            del state['_postgresDataset']

        return state
     
     
    def __del__(self) -> None:
        #self.closeAllConnections()
        pass
    
    # def worker_init_fn(self, worker_id: int) -> None:
    #     self._postgresDataset = PostgresDB(self._parametre)
    #     #print(f"Worker {worker_id} DB connection created")
        
    #     #print(f"Worker {worker_id}: {self._postgresDB}")
        
        
    # @staticmethod
    # def __synchronized(lock):
    #     def decorator(func):
    #         def wrapper(*args, **kwargs):
    #             with lock:
    #                 return func(*args, **kwargs)
    #         return wrapper
    #     return decorator

    
    def __createStream(self) -> None:
        """Crea una connessione solo se non è già presente per il processo."""
        
        if not hasattr(self, '_postgresDB') or self._postgresDB is None:
            self._postgresDB = PostgresDB(self._parametre)
       
            
    
    # @__synchronized(__lock)
    # def closeAllConnections(self) -> None:
    #     """Chiude la connessione per il processo specificato."""
        
    #     for stream in PostgresDB_Dataset.__processStream.values():
    #         stream.close_pool()
        
    #     PostgresDB_Dataset.__processStream.clear()
            
    # def getStream(self) -> PostgresDB:
    #     if self._postgresDB is None:
    #         self.__createStream()
        
    #     return self._postgresDB
    
    # @__synchronized(__lock)
    # def _getStream(self) -> PostgresDB:
    #     """Restituisce la connessione per il processo specificato."""
        
    #     processID = processID = process.current_process().pid
        
    #     if not self._isConnected(processID):
    #         self.__createStream(processID)
            
    #     return PostgresDB_Dataset.__processStream[processID] 
        
    #     return self._postgresDB     
        
