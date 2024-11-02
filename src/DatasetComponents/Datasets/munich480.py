from random import random
from matplotlib import pyplot as plt
import rasterio
from .ImageDataset import *
from .DatasetBase import *

import os
from os import listdir
from enum import Enum, auto, Flag
from typing import Tuple
import asyncio
import aiofiles

from threading import Lock
from functools import lru_cache
from functools import lru_cache

class Munich480(Segmentation_Dataset_Base):
    
    _EVAL_TILEIDS_FOLDER = os.path.join("tileids","val_folders.txt")
    _TRAIN_TILEIDS_FOLDER = os.path.join("tileids", "train_folders.txt")
    _TEST_TILEIDS_FOLDER = os.path.join("tileids", "test_folders.txt")
    _CLASSES_FILE = "classes.txt"
    
    _semaphore = asyncio.Semaphore(3)
    _classLock = Lock()
    _stack_axis: int = 0
    _classesMapping: dict = dict()
    _dataInitialized: bool = False
    _TemporalSize = 36
    _ImageChannels = 4
    _ImageWidth = 48
    _ImageHeight = 48
    
    class DataType(Enum):
        TRAINING = auto()
        TEST = auto()
        VALIDATION = auto()
        
    class Distance(Flag):
        m10 = auto()
        m20 = auto()
        m60 = auto()
        
    class Year(Flag):
        Y2016 = auto()
        Y2017 = auto()
        
    
    
    @staticmethod
    def _init_shared_data() -> None: 
        with Munich480._classLock:
            if Munich480._dataInitialized:
                return
            
            Munich480._read_classes()
            Munich480._dataInitialized = True
    
    @staticmethod
    def _read_classes() -> None:
        with open(Munich480._CLASSES_FILE, 'r') as f:
            classes = f.readlines()

        for row in classes:
            row = row.replace("\n", "")
            if '|' in row:
                id, cl = row.split('|')
                Munich480._classesMapping[int(id)] = cl
    
    @lru_cache(maxsize=27)
    @staticmethod
    def mapClassValue(index: int) -> str:
        assert index < 27 and index >= 0, "Index out of range"
        assert Munich480._dataInitialized, "Classes not initialized"
        return Munich480._classesMapping[index]
    



    def __new__(cls, *args, **kwargs):
        cls._init_shared_data()
        return super().__new__(cls)
           
  
    def __init__(self, folderPath:str | None, mode: DataType, year: Year, distance: Distance, transforms):
        
        assert type(mode) == Munich480.DataType, "Invalid mode type"
        assert type(year) == Munich480.Year, "Invalid year type"
        assert type(distance) == Munich480.Distance, "Invalid distance type"
        
        Segmentation_Dataset_Base.__init__(
            self, 
            imageSize = (Munich480._ImageWidth, Munich480._ImageHeight,Munich480._ImageChannels,Munich480._TemporalSize), 
            classesCount = 27, 
            transform=transforms,
            oneHot = False
        )
        
        temp_count = 0
        
        if Munich480.Distance.m10 in distance:
            temp_count += 1
        if Munich480.Distance.m20 in distance:
            temp_count += 1
        if Munich480.Distance.m60 in distance:
            temp_count += 1
        
        self._folderPath:str | None = folderPath
        self._years: Munich480.Year = year
        self._distance: Munich480.Distance = distance
        self._total_channels = Munich480._ImageChannels * Munich480._TemporalSize * temp_count
        
        match mode:
            case Munich480.DataType.TRAINING:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath,Munich480._TRAIN_TILEIDS_FOLDER), dtype=int)
                
            case Munich480.DataType.TEST:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath, Munich480._TEST_TILEIDS_FOLDER), dtype=int)
                
            case Munich480.DataType.VALIDATION:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath, Munich480._EVAL_TILEIDS_FOLDER), dtype=int)
                
        
        #print(self._dataSequenze)
    
    
    def _mapIndex(self, index: int) -> int:
        return self._dataSequenze[index]
    
    def get_dates(path: str, sample_number=None | int) -> list[str]:
        files = os.listdir(path)
        dates = list()
        for f in files:
            date = f.split("_")[0]
            if len(date) == 8:  # 20160101
                dates.append(date)

        dates = list(set(dates))
        
        if sample_number in None or sample_number < 0:
            return dates
        
        if len(dates) > sample_number:
            dates = random.sample(dates, sample_number)
        
        elif len(dates) < sample_number:
            while len(dates) < sample_number:
                dates.append(random.choice(dates))
        
        dates.sort()
        return dates
    
    @staticmethod
    def _normalize_dif_data(func) -> np.array:
        def wrapper(self, *args, **kwargs):
            data = func(self, *args, **kwargs)
            return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        return wrapper
    
    async def _normalize_dif_data(self, data: np.array) -> np.array:
        """Normalizza i dati tra 0 e 255."""
        return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
    
    #@_normalize_dif_data
    async def _load_dif_file(self, filePath:str, normalize: bool = True) -> np.array:
        #print(f"Loading: {filePath}")
        async with Munich480._semaphore:
            with rasterio.open(filePath) as src:
                data = src.read()
                
                if normalize:
                    return await self._normalize_dif_data(data)
                return data
                #return await self._normalize_dif_data(data)
        
    async def _load_y(self, folder:str):
        return await self._load_dif_file(filePath=os.path.join(folder, "y.tif"), normalize = False)
        
        
    async def _load_year_sequenze(self, year: str, number: str, use_coroutines: bool = True):
        sequenzeFolder = os.path.join(self._folderPath, year, number)
        tasks = []
        count = 0
        
        semaphore = asyncio.Semaphore(value=5)
        
        
        for file in os.listdir(sequenzeFolder):
            if file.endswith(("_10m.tif")):
                filePath=os.path.join(sequenzeFolder, file)
                tasks.append(self._load_dif_file(filePath))
                
                count += 1
                
                if count == 36:
                    break
                
        # Se ci sono meno di 36 file, replica gli ultimi caricati fino a raggiungere 36
        if count < 36:
            last_file = tasks[-1] if tasks else None  # Ultimo file caricato, se esiste
            while count < 36:
                if last_file:  
                    tasks.append(last_file)
                    count += 1
                
                
        
              

        if use_coroutines:
            tasks.append(self._load_y(sequenzeFolder))
            data_arrays = await asyncio.gather(*tasks)
            # Concatenate all data arrays along the last axis
            combined_data = np.concatenate(data_arrays[:-1], axis=Munich480._stack_axis)
            label = data_arrays[-1]
            return combined_data, label
        else:
            combined_data = None  # Inizializza a None
            for i, task in enumerate(tasks):
                npData = task()  # Assicurati che ogni task restituisca un array NumPy
                if combined_data is None:
                    combined_data = npData  # Primo array, inizializza combined_data
                else:
                    combined_data = np.concatenate((combined_data, npData), axis=Munich480._stack_axis)  # Concatenate i dati
            
            label = self._load_y(sequenzeFolder)  # Chiama _load_y senza await
            print(combined_data.shape, label.shape)
            return combined_data, label
        
        
                
    def _getSize(self) -> int:
        return np.size(self._dataSequenze)        

    async def _load_data(self, idx: int):
        data16: np.array | None = None
        data17: np.array | None = None
        label16: np.array | None = None
        label17: np.array | None = None
        
        if Munich480.Year.Y2016 in self._years:
            data16, label16 = await self._load_year_sequenze("data16", str(idx))
            return data16, label16    

        if Munich480.Year.Y2017 in self._years:
            data17, label17 = await self._load_year_sequenze("data17", str(idx))
            return data17, label17

    @DatasetBase._FormatResult
    def _getItem(self, idx: int) -> any:
        idx = self._mapIndex(idx)
 
        try:
            x, y = asyncio.run(self._load_data(idx))
            x = np.transpose(x, (1, 2, 0))
            y = np.squeeze(y, axis=0)
            return x, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return np.zeros(1, dtype=np.uint8)
        #return asyncio.run(_load_data())
            
        

    
    def visualize_sample(self, idx: int):
        """Visualizza il campione selezionato come immagine RGB e i singoli canali."""
        # Ottieni il campione usando _getItem
        try:
            data, label = asyncio.run(self._load_data(idx))
        except Exception as e:
            print(f"{e}")
            return

        # Verifica che data abbia 144 canali
        if data.shape[0] != 144:
            print("Il campione non ha 144 canali.")
            return

        # Riorganizza i dati in modo che ci siano 36 campioni di 4 canali
        data = data.reshape(36, 4, 48, 48)

        # Estrai i canali R, G, B, HV
        red_channel = data[:, 0]  # Canale R
        green_channel = data[:, 1]  # Canale G
        blue_channel = data[:, 2]  # Canale B
        hv_channel = data[:, 3]  # Canale HV

        # Combina i canali R, G e B per creare un'immagine RGB
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

        # Normalizza l'immagine RGB in [0, 255]
        rgb_image = np.clip(rgb_image, 0, 255)  # Clipping dei valori
        rgb_image = rgb_image.astype(np.uint8)  # Conversione in uint8

        # Configura il grafico
        fig, axs = plt.subplots(6, 36, figsize=(36, 6), constrained_layout=True)

        # Visualizza i singoli canali nelle prime 5 righe
        channels = [red_channel, green_channel, blue_channel, hv_channel]
        channel_names = ["R", "G", "B", "HV"]
        colormaps = ["Reds", "Greens", "Blues", "inferno"]  # Mappature dei colori per ogni canale
        
        for row in range(4):
            for col in range(36):
                axs[row, col].imshow(channels[row][col], cmap=colormaps[row])
                axs[row, col].axis("off")
                if col == 0:
                    axs[row, col].set_ylabel(channel_names[row])

        # Visualizza l'immagine RGB nell'ultima riga
        for col in range(36):
            axs[4, col].imshow(rgb_image[col])
            axs[4, col].axis("off")
        axs[4, 0].set_ylabel("RGB")

        # Mappa di colori per le etichette
        num_classes = 27
        cmap = plt.get_cmap('tab20', num_classes)  # Usa 'tab20' per colori distintivi

        # Rimuovi il primo asse da label per ottenere la forma (48, 48)
        label = label[0]  # Riduci la forma a (48, 48)

        # Visualizza le etichette nella sesta riga
        for col in range(36):
            axs[5, col].imshow(label, cmap='tab20', vmin=0, vmax=num_classes-1)
            axs[5, col].axis("off")
        axs[5, 0].set_ylabel("Label")

        plt.show()