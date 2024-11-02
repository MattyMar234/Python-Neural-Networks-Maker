from multiprocessing import process
from typing import Tuple

import numpy as np
from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import ImageTable, TableBase

from DatasetComponents.Datasets.DatasetBase import DatasetBase
from DatasetComponents.Datasets.DatasetBase import PostgresDB_Dataset

import torch
import time
import os
from PIL import Image
import pandas as pd
import ast
from .DatasetBase import DatasetBase


from abc import ABC, abstractmethod


class Segmentation_Dataset_Base(DatasetBase):
    
    def __init__(self, imageSize: Tuple[int, int, int, int] | Tuple[int, int, int], classesCount: int, x_transform, y_transform, oneHot: bool)-> None:
        DatasetBase.__init__(self, classesCount=classesCount, x_transform=x_transform, y_transform = y_transform, oneHot=oneHot, caching = False)
        
        assert imageSize is not None
        self.__img_Width = imageSize[0]
        self.__img_Height = imageSize[1]
        self.__img_Channels = imageSize[2]
        
        if len(imageSize) == 4:
            self.__img_TimeSequenze = imageSize[3]
    
        
        


class Row_Dataset_Base(ABC):
    pass

class Image_Dataset_Base(ABC):
    
    def __init__(self, imageSize: Tuple[int, int, int], stackChannel: bool = True)-> None:
        ABC.__init__(self)
        
        #print(f"class parametre: {imageSize}, {stackChannel}")
        
        assert imageSize is not None
        
        #self._imageSize: Tuple[int, int, int] = imageSize
        self.__stackChannel = stackChannel
        self.__img_Width = imageSize[0]
        self.__img_Height = imageSize[1]
        self.__img_Channels = imageSize[2]
    
    @staticmethod
    def _processImageData(func) -> np.array:
        def wrapper(self, *args, **kwargs):
            
            r_channel: np.array = None
            g_channel: np.array = None
            b_channel: np.array = None
            data: np.array = None
            label:int = None
            
            r_channel, g_channel, b_channel, label = func(self, *args, **kwargs)
            
            r_channel = r_channel.reshape(self.__img_Width, self.__img_Height)
            g_channel = g_channel.reshape(self.__img_Width, self.__img_Height)
            b_channel = b_channel.reshape(self.__img_Width, self.__img_Height)
            
            #print(r_channel, g_channel, b_channel)
            # print(r_channel.shape, g_channel.shape, b_channel.shape)
            
            
            rgb_array = (np.stack((r_channel, g_channel, b_channel), axis=-1))
            image = Image.fromarray(rgb_array)
            
            
            
            
            # if self.__stackChannel:
            #     data = (np.stack((r_channel, g_channel, b_channel), axis=-1)) / 255
            #     #Ho il formato (28, 28, 3). Ma Pytorch vuole (3, 28, 28)
            #     # Trasporre i canali per avere (3, 28, 28) anziché (28, 28, 3)
            #     data = np.transpose(data, (2, 0, 1))
            # else:
            #     data = np.concatenate((r_channel, g_channel, b_channel)) / 255
        
            return image, label
        return wrapper
    
class ImageDataset_CSV(Image_Dataset_Base, DatasetBase):
    
    def __init__(self, file: str, imageSize: Tuple[int, int, int], classesCount: int, transform, oneHot: bool, stackChannel: bool)-> None:
        Image_Dataset_Base.__init__(self, imageSize=imageSize, stackChannel = stackChannel )
        DatasetBase.__init__(self, classesCount=classesCount, transform=transform, oneHot=oneHot)
        
        self.__stream = pd.read_csv(file, header=0)
        self._dataset_size = len(self.__stream)
        self.__imgSize = imageSize
        
    def _getSize(self) -> int:
        return self._dataset_size 
    
    
    def __read_row(self, file_stream, idx) -> tuple:
           
            result = file_stream.iloc[idx].to_numpy(dtype=np.uint8)
            arrayLenght = self.__imgSize[0] * self.__imgSize[1]
            r_start = 0
            r_end = r_start + arrayLenght
            g_start = r_end
            g_end = g_start + arrayLenght
            b_start = g_end
            b_end = b_start + arrayLenght

            label = 0#result[0]
            r_channel = result[r_start:r_end].reshape(self.__imgSize[0], self.__imgSize[1])
            g_channel = result[g_start:g_end].reshape(self.__imgSize[0], self.__imgSize[1])
            b_channel = result[b_start:b_end].reshape(self.__imgSize[0], self.__imgSize[1])
            
            return r_channel, g_channel, b_channel, label
        
    @DatasetBase._FormatResult
    @Image_Dataset_Base._processImageData
    def _getItem(self, idx: int) -> any:
        
        if idx >= self._dataset_size or idx < 0:
            raise IndexError("Index out of range")
        
        return self.__read_row(self.__stream, idx)
    

class ImageDataset_CSV_form_POSTGRES(Image_Dataset_Base, DatasetBase):
    
    def __init__(self, file: str, imageSize: Tuple[int, int, int], classesCount: int, transform, oneHot: bool, stackChannel: bool)-> None:
        Image_Dataset_Base.__init__(imageSize=imageSize, stackChannel = stackChannel )
        DatasetBase.__init__(self, classesCount=classesCount, transform=transform, oneHot=oneHot)
        
        self.__stream = pd.read_csv(file, header=0)
        self.__dataset_size = len(self.__b_stream)
        
    
    def _getSize(self) -> int:
        self.__dataset_size 
    
    
    @DatasetBase._FormatResult
    @Image_Dataset_Base._processImageData
    def _getItem(self, idx: int) -> any:
        
        if idx >= self.__dataset_size or idx < 0:
            raise IndexError("Index out of range")
        
        result = self.__stream.iloc[idx]
        
        dict_red_channel  = result['red_channel']
        dict_green_channel  = result['green_channel']
        dict_blue_channel  = result['blue_channel']
        label = label = int(result['label']) 
        
        
        database: PostgresDB = self._getStream()
        result = database.fetch_results(self.__table.getElementAt(index=idx))[0]
        
        #print(result)
        
        label: int = result[1]
        r_channel: np.array = np.array(result[2], dtype=np.uint8)
        g_channel: np.array = np.array(result[3], dtype=np.uint8)
        b_channel: np.array = np.array(result[4], dtype=np.uint8)
    

        return r_channel, g_channel, b_channel, label
    


class ImageDataset_Postgres(Image_Dataset_Base, PostgresDB_Dataset):
    
    def __init__(self, imageSize: Tuple[int, int, int], classesCount: int, connectionParametre: DatabaseParametre, table: ImageTable , transform, oneHot: bool, stackChannel: bool):
        
        assert (table is not None) 
        
        
        Image_Dataset_Base.__init__(
            self, 
            imageSize=imageSize, 
            stackChannel = stackChannel
        )
        
        PostgresDB_Dataset.__init__(
            self, 
            classesCount= classesCount, 
            connectionParametre=connectionParametre, 
            transform=transform, 
            oneHot = oneHot
        )
    
        self.__table: ImageTable = table
        
    def _getSize(self) -> int:
        return self._getStream().fetch_results(self.__table.getTableElementsCount())[0][0]
    
    
    @DatasetBase._FormatResult
    @Image_Dataset_Base._processImageData
    def _getItem(self, idx: int) -> any:
        
        database: PostgresDB = self._getStream()
        
        try:
            query = self.__table.getElementAt(index=idx)
            result = database.fetch_results(query)[0]
        
        except Exception as e:
            print(e)
            print("idx: ", idx)
            print("query: ", query)
            raise e
        
        #print(result)
        
        label: int = result[1]
        r_channel: np.array = np.array(result[2], dtype=np.uint8)
        g_channel: np.array = np.array(result[3], dtype=np.uint8)
        b_channel: np.array = np.array(result[4], dtype=np.uint8)
    

        return r_channel, g_channel, b_channel, label
        
        #return torch.Tensor([1,2,3]), torch.Tensor([1])
     