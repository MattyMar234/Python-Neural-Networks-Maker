from ast import Tuple
from functools import lru_cache
from typing import Dict, Final, List
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
import torch
import opendatasets
import colorsys
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from DatasetComponents.Datasets.munich480 import Munich480
import Globals

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os



def _generate_distinct_colors(num_classes: int) -> Dict[int, str]:
    colors = {}
    hue_step = 1 / num_classes  # Ampio intervallo per distribuire i colori distintamente
    
    for i in range(num_classes):
        # Genera un hue con un intervallo ampio per garantire differenze cromatiche
        hue = (i * hue_step) % 1.0
        # Variamo leggermente la luminosità e la saturazione per evitare colori simili
        lightness = 0.5 + (i % 2) * 0.1  # Alterna tra due valori di luminosità
        saturation = 0.8 - (i % 3) * 0.2  # Alterna tre valori di saturazione

        # Converti HLS a RGB e poi in esadecimale
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        colors[i] = hex_color
    
    return colors

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)


class Munich480_DataModule(DataModuleBase):
    
    TemporalSize: Final[int] = 32
    ImageChannels: Final[np.array] = np.array([4,6,3])
    ImageWidth: Final[int] = 48
    ImageHeight: Final[int] = 48
    ClassesCount: Final[int] = 27
    
    _SINGLETON_INSTANCE = None
    
    
    _KAGGLE_DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480"
    _CLASSES_FILE = "classes.txt"
    # _MAP_COLORS: Dict[int, str] = {
    #     0 : '#000000',  1 : '#ff7f0e',  2 : '#2ca02c',   3 : '#d62728',
    #     4 : '#9467bd',  5 : '#8c564b',  6 : '#e377c2',   7 : '#7f7f7f',
    #     8 : '#bcbd22',  9 : '#17becf', 10 : '#ff9896',  11 : '#98df8a',
    #     12 : '#c5b0d5', 13 : '#ffbb78', 14 : '#c49c94',  15 : '#f7b6d2',
    #     16 : '#9edae5', 17 : '#aec7e8', 18 : '#ffcc5c',  19 : '#ff6f69',
    #     20 : '#96ceb4', 21 : '#ff9b85', 22 : '#c1c1c1',  23 : '#ffd700',
    #     24 : '#b2e061', 25 : '#ff4f81', 26 : '#aa42f5' 
    # }
    
    MAP_COLORS: Dict[int, str] = _generate_distinct_colors(27)
    MAP_COLORS_AS_RGB_LIST: Dict[int, List[int]] = {key: hex_to_rgb(value) for key, value in MAP_COLORS.items()}

    
    
    
    def __setstate__(self, state):
        return
     
    def __getstate__(self) -> object:
        return {}
    
    #singleton
    def __new__(cls, *args, **kwargs):
        if cls._SINGLETON_INSTANCE is None:
            cls._SINGLETON_INSTANCE = super().__new__(cls)
        return cls._SINGLETON_INSTANCE
    
    
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
        self._setup_done = False
        
        self._year = year
        self._persistent_workers: bool = True
        self._pin_memory: bool = True
        self._useTemporalSize = useTemporalSize
        self._total_channel = 13
        self._prefetch_factor: int | None = 1
        
        
        self._classesMapping: dict = {}
        self._read_classes()
        
        if self._num_workers == 0:
            self._persistent_workers = False
            self._pin_memory = False
            self._prefetch_factor = None
        
            
        if not useTemporalSize:
            self._total_channel *= Munich480_DataModule.TemporalSize
        
        
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
            return [1, self._total_channel, Munich480_DataModule.TemporalSize, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]
        else: 
            return [1, self._total_channel, Munich480_DataModule.ImageHeight, Munich480_DataModule.ImageWidth]

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
        #print(self._classesMapping)
        
    
    def map_classes(self, classes: np.ndarray | List[int] | int) -> List[str] | str | None:
        
        if type(classes) == int:
            return f"{self._classesMapping[classes]} - {classes}"
        
        if type(classes) == np.ndarray:
            classes = classes.tolist()
        
        list_classes: List[str] = []
          
        for i in classes:
            list_classes.append(f"{self._classesMapping[i]} - {i}")
        
        return list_classes
        
    def classesToIgnore(self) -> List[int]:
        sequenze = np.arange(0, self.ClassesCount)
        ingnore = []
        
        assert len(self._classesMapping.keys()) > 0, "self._classesMapping is empty"
        
        for i in sequenze:
            if i not in self._classesMapping.keys():
                ingnore.append(i)
           
        
        return ingnore
        
        
    def calculate_classes_weight(self) ->torch.tensor:
        if not self._setup_done:
            self.setup()
            
        core: int | None = os.cpu_count()
        
        if core == None:
            core = 0
            batch_size = 4
        else:
            batch_size:int = core * 20
        
        self._TRAIN.setLoadOnlyY(True)
        
        temp_loader = DataLoader(
            self._TRAIN,  # Assuming self._TRAIN is a Dataset object
            batch_size=10,  # Set batch_size=1 for individual sample processing
            num_workers=core,
            shuffle=False,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=None
        )
        
        Globals.APP_LOGGER.info("Calulating classes weight...")
        
        y_flat: torch.Tensor = torch.empty((len(self._TRAIN) * Munich480_DataModule.ImageWidth * Munich480_DataModule.ImageHeight), dtype=torch.uint8)
        index: int = 0
    
        for i, (_, y) in enumerate(temp_loader):
            print(f"Loading: {min(i + 1, len(temp_loader))}/{len(temp_loader)}", end="\r")
            
            y = y.flatten()
            elementCount = y.shape[0]
            y_flat[index: index + elementCount] = y
            index += elementCount
            
        print()
    
            
        # print("my weights")
        # class_counts: torch.Tensor = torch.zeros(self.output_classes)# + 1e-12
        # class_counts += torch.bincount(y_flat, minlength=self.output_classes)

        # # Calcola i pesi come inverso della frequenza, e imposta a zero le classi assenti
        # weights = torch.zeros(self.output_classes)
        # weights[class_counts > 0] = 1.0 / class_counts[class_counts > 0].float()
        # weights = weights / weights.sum()  # Normalizza i pesi per sommare a 1
        
        # weightsDict: Dict[int, float] = {i: weights[i].item() for i in range(len(weights))}

        # for i in range(len(weights)):
        #     try:
        #         print(f"{weightsDict[i]:.10f} -> {self.map_classes(i)}")
        #     except:
        #         print(f"{weightsDict[i]:.10f} -> ?")
        
        
        y_flat = y_flat.numpy()
        available_classes = np.unique(y_flat)
        
        class_weights = compute_class_weight(
            class_weight='balanced',  # Opzione per bilanciare in base alla frequenza
            classes=available_classes,  # Array di tutte le classi
            y=y_flat
        )
        
        # Mappa i pesi su tutte le classi, assegnando peso 0 alle classi assenti
        weights = np.zeros(Munich480_DataModule.ClassesCount, dtype=np.float32)
        weights[available_classes] = class_weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        Globals.APP_LOGGER.info(f"Calculated classes weight: {weights_tensor}")

        self._TRAIN.setLoadOnlyY(False)
        return weights_tensor


    def setup(self, stage=None) -> None:
        if self._setup_done:
            return
        
        self._TRAIN = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.TRAINING, year= self._year, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        self._VAL   = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.VALIDATION, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._TEST  = Munich480(self._datasetFolder, mode= Munich480.DatasetMode.TEST, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._setup_done = True


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        #return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    
   
    
    
    def show_processed_sample(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, index: int, confusionMatrixData: Dict[str, any], X_as_Int: bool = False, temporalSequenze = False) -> None:
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
        fig1.suptitle(f"Index {index}")
        
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
        color_list = [color for _, color in sorted(Munich480_DataModule.MAP_COLORS.items())]
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
        legend_patches = [mpatches.Patch(color=Munich480_DataModule.MAP_COLORS[cls], label=f'{cls} - {label}') for cls, label in self._classesMapping.items()]
        fig2.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc='upper right', title="Class Colors")

        fig1.subplots_adjust(right=0.8)
        fig2.subplots_adjust(right=0.7)
        
        fig1.tight_layout(pad=2.0)
        
        # fig3, axes3 = plt.subplots(1, 1, figsize=(10, 5))
        # axes3.imshow(confusionMatrix, cmap='hot')
        
        # Finestra 3: Matrice di confusione
        
        
        plt.figure("Confusion Matrix", figsize=(8, 6))
        sns.heatmap(
            data=confusionMatrixData['data'], 
            annot=confusionMatrixData['annot'], 
            fmt=confusionMatrixData['fmt'], 
            cmap=confusionMatrixData['cmap'], 
            cbar=confusionMatrixData['cbar'],
            xticklabels=confusionMatrixData['xticklabels'], 
            yticklabels=confusionMatrixData['yticklabels']
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        
        #plt.imshow(confusionMatrix)
        plt.show()
        
        

        
        
        
        