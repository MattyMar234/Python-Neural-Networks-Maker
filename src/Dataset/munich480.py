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

class Munich480(Segmentation_Dataset_Base):
    
    _EVAL_TILEIDS_FOLDER = os.path.join("tileids","val_folders.txt")
    _TRAIN_TILEIDS_FOLDER = os.path.join("tileids", "train_folders.txt")
    _TEST_TILEIDS_FOLDER = os.path.join("tileids", "test_folders.txt")
    
    
    class DataType(Enum):
        TRAINING = auto()
        TEST = auto()
        VALIDATION = auto()
        
    class Year(Flag):
        Y2016 = auto()
        Y2017 = auto()
        
    
    __stack_axis = 0
    
    
    def __init__(self, folderPath:str | None, mode: DataType, year: Year, transforms):
        Segmentation_Dataset_Base.__init__(self, imageSize = (48,48,4,30), classesCount = 27, transform=transforms, oneHot = True)
        
        self._folderPath:str | None = folderPath
        self._years: Munich480.Year = year
    
        
        match mode:
            case Munich480.DataType.TRAINING:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath,Munich480._TRAIN_TILEIDS_FOLDER), dtype=int)
                
            case Munich480.DataType.TEST:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath, Munich480._TEST_TILEIDS_FOLDER), dtype=int)
                
            case Munich480.DataType.VALIDATION:
                self._dataSequenze = np.loadtxt(os.path.join(self._folderPath, Munich480._EVAL_TILEIDS_FOLDER), dtype=int)
                
        
        print(self._dataSequenze)
    
    
    def _mapIndex(self, index: int) -> int:
        return self._dataSequenze[index]
    
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
        
        with rasterio.open(filePath) as src:
            data = src.read()
            
            if normalize:
                return await self._normalize_dif_data(data)
            return data
            #return await self._normalize_dif_data(data)
        
    async def _load_y(self, folder:str):
        return await self._load_dif_file(filePath=os.path.join(folder, "y.tif"), normalize = False)
        
        
    async def _load_year_sequenze(self, year: str, number: str):
        sequenzeFolder = os.path.join(self._folderPath, year, number)
        tasks = []
        count = 0
        
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
                
                
        tasks.append(self._load_y(sequenzeFolder))
              
                #self._load_dif_file(filePath=os.path.join(sequenzeFolder, file))
        # Wait for all tasks to complete
        data_arrays = await asyncio.gather(*tasks)
        
        # Concatenate all data arrays along the last axis
        combined_data = np.concatenate(data_arrays[:-1], axis=Munich480.__stack_axis)
        label = data_arrays[-1]
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
            return asyncio.run(self._load_data(idx))
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