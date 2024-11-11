from ast import Tuple
from typing import Dict, Final
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
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
    _CLASSES_FILE = "classes.txt"
    _MAP_COLORS: Dict[int, str] = {
        0 : '#000000',  1 : '#ff7f0e',  2 : '#2ca02c',   3 : '#d62728',
        4 : '#9467bd',  5 : '#8c564b',  6 : '#e377c2',   7 : '#7f7f7f',
        8 : '#bcbd22',  9 : '#17becf', 10 : '#ff9896',  11 : '#98df8a',
        12 : '#c5b0d5', 13 : '#ffbb78', 14 : '#c49c94',  15 : '#f7b6d2',
        16 : '#9edae5', 17 : '#aec7e8', 18 : '#ffcc5c',  19 : '#ff6f69',
        20 : '#96ceb4', 21 : '#ff9b85', 22 : '#c1c1c1',  23 : '#ffd700',
        24 : '#b2e061', 25 : '#ff4f81', 26 : '#aa42f5' 
    }
    
    def __setstate__(self, state):
        return
     
    def __getstate__(self) -> object:
        return {}
    
    
    def __init__(
        self, 
        datasetFolder:str, 
        year: Munich480.Year = Munich480.Year.Y2016, 
        download: bool = False, 
        batch_size: int = 1, 
        num_workers: int  = 1,
        useTemporalSize: bool = False
    ):
        super().__init__(datasetFolder, batch_size, num_workers)

        assert os.path.exists(datasetFolder), f"La cartella {datasetFolder} non esiste"
        assert type(year) == Munich480.Year, f"year deve essere di tipo Munich480.Year"
        
        
        self._download = download
        self._TRAIN: Munich480 | None = None
        self._VAL: Munich480 | None = None
        self._TEST: Munich480 | None = None
        
        self._year = year
        self._persistent_workers: bool = True
        self._pin_memory: bool = True
        self._useTemporalSize = useTemporalSize
        self._total_channel = 13
        self._prefetch_factor: int | None = 1
        
        self._classesMapping: dict = {}
        
        if self._num_workers == 0:
            self._persistent_workers = False
            self._pin_memory = False
            self._prefetch_factor = None
        
            
        if not useTemporalSize:
            self._total_channel *= Munich480_DataModule.TemporalSize * (len(year))
        
        
        self._training_trasforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
            #transforms.Resize((572, 572)),
            #transforms.ToTensor() 
        ])
        
        self._test_trasforms = transforms.Compose([
            #transforms.Resize((572, 572)),
            #transforms.ToTensor()
        ])
    
    @property
    def input_channels(self) -> int:
        return self._total_channel
    
    @property
    def output_classes(self) -> int:
        return Munich480_DataModule.ClassesCount
    
    @property    
    def input_size(self) -> list[int]:
        if self._useTemporalSize:
            return [1, Munich480_DataModule.TemporalSize, self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]
        else: 
            return [1,self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]

    def prepare_data(self) -> None:
        if self._download:
            self._DownloadDataset(url= Munich480_DataModule._KAGGLE_DATASET_URL, folder= self._datasetFolder)
    
    def _read_classes(self) -> None:
        with open(os.path.join(self._datasetFolder, Munich480_DataModule._CLASSES_FILE), 'r') as f:
            classes = f.readlines()

        for row in classes:
            row = row.replace("\n", "")
            if '|' in row:
                id, cl = row.split('|')
                self._classesMapping[int(id)] = cl     
        print(self._classesMapping)


    def setup(self, stage=None) -> None:
        
        self._read_classes()
        
        self._TRAIN = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.TRAINING, year= self._year, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        self._VAL   = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.VALIDATION, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._TEST  = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.TEST, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)



    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        #return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def show_processed_sample(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, index: int, X_as_Int: bool = False, temporalSequenze = False) -> None:
        assert x is not None, "x is None"
        assert y_hat is not None, "y_hat is None"
        assert y is not None, "y is None"
        
        
        
        x = x.cpu().detach()
        x = x.squeeze(0) # elimino la dimensione della batch
        
        if temporalSequenze:
            x = x.permute(1, 0, 2, 3)
            x = x.reshape(-1, 48, 48)
            
       
        
        if X_as_Int:
            x = x.int()
        else :
            x = ((x * (2**16 - 1))*1e-4)
            x *= 255
            x = x.int()
            x += 40
            x = x.clamp(0, 255)
            
        
        y_hat = y_hat.cpu().detach().squeeze(0)
        y = y.cpu().detach()
        
        
        #fig, axes = plt.subplots(rowElement, (num_images // rowElement) + (num_images % rowElement != 0) + 3, figsize=(16, 12))  # Griglia verticale per ogni immagine

        

        #=========================================================================#
        # Prima finestra: Visualizzazione delle immagini RGB
        rowElement: int = 8
        num_images: int = int(x.shape[0] // 13)  # Numero di immagini nel batch
        num_cols: int = (num_images // rowElement) + (num_images % rowElement != 0)
        
        fig1, axes1 = plt.subplots(rowElement, num_cols, figsize=(10, 12))
        fig1.suptitle(f"Images index {index}")
        
        for idx in range(num_images):
            row: int = idx % rowElement
            col: int = idx // rowElement
            
            # Estrazione dell'immagine (13 canali)
            image = x[idx * 13:(idx + 1) * 13, :, :]
            red, green, blue = image[2], image[1], image[0]
            rgb_image = torch.stack([red, green, blue], dim=0).permute(1, 2, 0).numpy()

            ax = axes1[row, col]
            ax.imshow(rgb_image)
            ax.set_title(f"Image {idx+1}")
            ax.axis('off')
        
        
        # Crea una colormap personalizzata
        color_list = [color for _, color in sorted(Munich480_DataModule._MAP_COLORS.items())]
        cmap = ListedColormap(color_list)
        
        label_map = y.argmax(dim=0).numpy()         # Etichetta per l'immagine corrente
        pred_map = y_hat.argmax(dim=0).numpy()      # Predizione con massimo di ciascun layer di `y_hat`
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle("Feature Maps: Label Map and Prediction Map")
        
            # Mappa etichetta `y`
        axes2[0].imshow(label_map, cmap=cmap)
        axes2[0].set_title("Label Map")
        axes2[0].axis('off')
        
        # Mappa predizione `y_hat`
        axes2[1].imshow(pred_map, cmap=cmap)
        axes2[1].set_title("Prediction Map")
        axes2[1].axis('off')
        
        # Aggiungi legenda accanto alla seconda figura
        legend_patches = [mpatches.Patch(color=Munich480_DataModule._MAP_COLORS[cls], label=f'{cls} - {label}') for cls, label in self._classesMapping.items()]
        fig2.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc='upper right', title="Class Colors")

        fig1.subplots_adjust(right=0.8)
        fig2.subplots_adjust(right=0.7)
        
        fig1.tight_layout(pad=2.0)
        plt.show()
        
        
        # for i in range(8):
           
        #     # 2. Mappa etichetta `y`
        #     ax = axes[i, col]
        #     ax.imshow(label_map, cmap=cmap)
        #     #ax.set_title(f"Label {i+1}")
        #     ax.axis('off')
            
        #     # 3. Mappa predizioni `y_hat`
        #     ax = axes[i, col + 1]
        #     ax.imshow(pred_map, cmap=cmap)
        #     #ax.set_title(f"Pred. {i+1}")
        #     ax.axis('off')
            
        #     # 4. Mappa della loss
        #     # ax = axes[3, i]
        #     # ax.imshow(loss_map, cmap='hot')
        #     # ax.set_title(f"Loss {i+1}")
        #     # ax.axis('off')
            
        # # Aggiungi la legenda con la corrispondenza colore-categoria
        # legend_patches = [mpatches.Patch(color=Munich480_DataModule._MAP_COLORS[cls], label=f'{label}') for cls, label in self._classesMapping.items()]#for cls, color in Munich480_DataModule._MAP_COLORS.items()]
        # plt.legend(handles=legend_patches, bbox_to_anchor=(1.2, 0.5), loc='upper left', borderaxespad=0.)

        # plt.tight_layout(pad=1.0)
        # plt.subplots_adjust(right=0.50)
        # plt.show()
        
        
        
        