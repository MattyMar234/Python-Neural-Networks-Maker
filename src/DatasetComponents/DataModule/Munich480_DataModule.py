from ast import Tuple
from typing import Final
from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
import torch
import opendatasets

from DatasetComponents.Datasets.munich480 import Munich480

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os
import os
import os
#from DatasetComponents.Datasets.munich480 import *


class Munich480_DataModule(DataModuleBase):
    
    TemporalSize: Final[int] = 32
    ImageChannels: Final[np.array] = np.array([4,6,3])
    ImageWidth: Final[int] = 48
    ImageHeight: Final[int] = 48
    
    ClassesCount: Final[int] = 27
    
    _KAGGLE_DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480"
    
    def __init__(
        self, 
        datasetFolder:str, 
        distance: Munich480.Distance = Munich480.Distance.m10, 
        year: Munich480.Year = Munich480.Year.Y2016, 
        download: bool = False, 
        batch_size: int = 1, 
        num_workers: int  = 1,
        useTemporalSize: bool = False
    ):
        super().__init__(datasetFolder, batch_size, num_workers)

        assert os.path.exists(datasetFolder), f"La cartella {datasetFolder} non esiste"
        assert type(distance) == Munich480.Distance, f"distance deve essere di tipo Munich480.Distance"
        assert type(year) == Munich480.Year, f"year deve essere di tipo Munich480.Year"
        
        
        self._download = download
        self._TRAIN: Munich480 | None = None
        self._VAL: Munich480 | None = None
        self._TEST: Munich480 | None = None
        
        self._distance = distance
        self._year = year
        self._persistent_workers: bool = True
        self._pin_memory: bool = True
        self._useTemporalSize = useTemporalSize
        self._total_channel = 0
        
        if Munich480.Distance.m10 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[0]
        if Munich480.Distance.m20 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[1]
        if Munich480.Distance.m60 in distance:
            self._total_channel += Munich480_DataModule.ImageChannels[2]
            
        if not useTemporalSize:
            self._total_channel *= Munich480_DataModule.TemporalSize * (len(year))
        
        
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
        
    def number_of_channels(self) -> int:
        return self._total_channel
    
    def number_of_classes(self) -> int:
        return Munich480_DataModule.ClassesCount
        
    def input_size(self) -> list[int]:
        if self._useTemporalSize:
            return [1, Munich480_DataModule.TemporalSize, self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]
        else: 
            return [1,self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]

    def prepare_data(self) -> None:
        if self._download:
            self._DownloadDataset(url= Munich480_DataModule._KAGGLE_DATASET_URL, folder= self._datasetFolder)
    

    def setup(self, stage=None) -> None:
        self._TRAIN = Munich480(self._datasetFolder, mode= Munich480.DataType.TRAINING, year= self._year, distance=self._distance, transforms=self._training_trasforms)
        self._VAL   = Munich480(self._datasetFolder, mode= Munich480.DataType.VALIDATION, year= self._year, distance=self._distance, transforms=self._test_trasforms)
        self._TEST  = Munich480(self._datasetFolder, mode= Munich480.DataType.TEST, year= self._year, distance=self._distance, transforms=self._test_trasforms)



    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True, prefetch_factor=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True, prefetch_factor=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=(self._persistent_workers and (self._num_workers > 0)), drop_last=True, prefetch_factor=1)
    
    
    def show_processed_sample(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, X_as_Int: bool = False) -> None:
        assert x is not None, "x is None"
        assert y_hat is not None, "y_hat is None"
        assert y is not None, "y is None"
        
        rowElement: int = 8
        col:int = 0
        row: int = 0
        idx: int = 0
        
        
        x = x.cpu().detach()
        x = x.squeeze(0) # elimino la dimensione della batch
        
        if X_as_Int:
            x = x.int()
        else :
            x = x * 255
            x = x.int()
        
        
        y_hat = y_hat.cpu().detach().squeeze(0)
        y = y.cpu().detach()
        
        num_images = int((x.shape[0] // 13))  # Numero di immagini nel batch
        fig, axes = plt.subplots(rowElement, (num_images // rowElement) + (num_images % rowElement != 0) + 3, figsize=(16, 12))  # Griglia verticale per ogni immagine

        label_map = y.argmax(dim=0).numpy()         # Etichetta per l'immagine corrente
        pred_map = y_hat.argmax(dim=0).numpy()      # Predizione con massimo di ciascun layer di `y_hat`

        
        while idx < num_images:
        
            # Estrazione dell'immagine (13 canali)
            image = x[idx*13:(idx+1)*13, :, :]
            red, green, blue = image[2], image[1], image[0]
            rgb_image = torch.stack([red, green, blue], dim=0).permute(1, 2, 0).numpy()

           
            ax = axes[row, col]
            ax.imshow(rgb_image)
            #ax.set_title(f"Image {i+1} RGB")
            ax.axis('off')
            
            row += 1
            idx += 1
            
            if row == rowElement:
                row = 0
                col += 1
                
            
        
        if (num_images % rowElement != 0):       
            col += 1
        
        for i in range(8):
           
            # 2. Mappa etichetta `y`
            ax = axes[i, col]
            ax.imshow(label_map, cmap='tab20')
            #ax.set_title(f"Label {i+1}")
            ax.axis('off')
            
            # 3. Mappa predizioni `y_hat`
            ax = axes[i, col + 1]
            ax.imshow(pred_map, cmap='tab20')
            #ax.set_title(f"Pred. {i+1}")
            ax.axis('off')
            
            # 4. Mappa della loss
            # ax = axes[3, i]
            # ax.imshow(loss_map, cmap='hot')
            # ax.set_title(f"Loss {i+1}")
            # ax.axis('off')

        plt.tight_layout()
        plt.show()
        
        
        
        