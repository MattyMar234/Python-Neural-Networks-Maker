from abc import ABC, abstractmethod
import numpy as np

__all__ = ["TableBase", "TrainingImages", "TestImages"]

class TableBase(ABC):
    
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def createTableQuery(self) -> str:
        pass
    
    @abstractmethod
    def generateInsertQuery(self, *args, **kwargs) -> str:
        pass
    
    @abstractmethod
    def getTableName(self) -> str:
        pass

    @abstractmethod
    def dropTableQuery(self) -> str:
        pass



class ImageTable(TableBase):
    
    __ImageSize = 32*32
    
    def __init__(self, tableName) -> None:
        super().__init__()

        self._table_name = f"{tableName}_CIFAR10"
        self._id_column = "id"
        self._label_column = "label"
        self._rChannel_column = "red_channel"
        self._gChannel_column = "green_channel"
        self._bChannel_column = "blue_channel"


    def getTableName(self) -> str:
        return self._table_name
    
    def createTableQuery(self) -> str:
        return f"CREATE TABLE IF NOT EXISTS {self._table_name} ( " +\
                f"{self._id_column} SERIAL PRIMARY KEY, " +\
                f"{self._label_column} INT NOT NULL, " +\
                f"{self._rChannel_column} INT[{ImageTable.__ImageSize}] NOT NULL, " +\
                f"{self._gChannel_column} INT[{ImageTable.__ImageSize}] NOT NULL, " +\
                f"{self._bChannel_column} INT[{ImageTable.__ImageSize}] NOT NULL);"
    
    def getElementAt(self, index: int, order: bool = False) -> str:
        # if order:
        #     return f"SELECT * FROM {self._table_name} ORDER BY {self._id_column} LIMIT 1 OFFSET {index};"
        # else:
        return f"SELECT * FROM {self._table_name} WHERE {self._id_column} = {index + 1};"# LIMIT 1;"
        
    def dropTableQuery(self) -> str:
        return f"DROP TABLE IF EXISTS {self._table_name};"
    
    def getTableElementsCount(self) -> str:
        return f"SELECT COUNT(*) FROM {self._table_name};"
    
    def generateInsertQuery(self, label: str, rCh: np.array, gCh: np.array, bCh: np.array) -> str:
        return f"INSERT INTO {self._table_name} ({self._label_column}, " +\
                f"{self._rChannel_column}, {self._gChannel_column}, " +\
                f"{self._bChannel_column}) VALUES ({label}, ARRAY{rCh.tolist()}, ARRAY{gCh.tolist()}, ARRAY{bCh.tolist()});"
        
    

class TrainingImages(ImageTable):

    __instance = None
   
    #singleton
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(TrainingImages, cls).__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        super().__init__(type(self).__name__)


class TestImages(ImageTable):

    __instance = None
   
    #singleton
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(TestImages, cls).__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        super().__init__(type(self).__name__)
         