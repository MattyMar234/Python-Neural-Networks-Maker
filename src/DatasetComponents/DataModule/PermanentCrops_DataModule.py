import pickle
from typing import Dict, Final, List
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
from psycopg2 import Binary
import pytorch_lightning as pl
from torchvision import transforms
import torch
import opendatasets
import colorsys
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import TableBase, TensorTable
from DatasetComponents.Datasets.DatasetBase import PostgresDataset_Interface
from DatasetComponents.Datasets.PermanentCrops import PermanentCrops
from DatasetComponents.Datasets.munich480 import Munich480
import Globals
from Networks.Metrics.ConfusionMatrix import ConfusionMatrix
from Networks.NetworkComponents.NeuralNetworkBase import *
from Utility.TIF_creator import TIF_Creator

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os
import time

#https://content.satimagingcorp.com/media2/filer_public_thumbnails/filer_public/44/9c/449caa01-64b9-417f-9547-964b66465554/cms_page_media1530image001.png__525.0x426.0_q85_subsampling-2.jpg
class BandeIndex(Enum):
    BAND_1_AEROSOL: int = 10
    BAND_2_RED: int = 0
    BAND_3_GREEN: int = 1
    BAND_4_BLUE: int = 2
    BAND_5_VRE1: int = 4
    BAND_6_VRE2: int = 5
    BAND_7_VRE3: int = 6
    BAND_8_NIR: int = 3
    BAND_A8: int = 7
    BAND_9_WV: int = 11
    BAND_10_SWIR: int = 12
    BAND_11_SWIR1: int = 8
    BAND_12_SWIR2: int = 9
    
# Natural Color (B4, B3, B2) ...
# Color Infrared (B8, B4, B3) ...
# Short-Wave Infrared (B12, B8A, B4) ...
# Agriculture (B11, B8, B2) ...
# Geology (B12, B11, B2) ...
# Bathymetric (B4, B3, B1) ...
# Vegetation Index (B8-B4)/(B8+B4) ...
# Moisture Index (B8A-B11)/(B8A+B11)

class PermanentCrops_DataModule(DataModuleBase):
    
    TemporalSize: Final[int] = 62
    ImageChannels: Final[int] = 13
    ImageWidth: Final[int] = 48*2
    ImageHeight: Final[int] = 48*2
    ClassesCount: Final[int] = 4
    
    _SINGLETON_INSTANCE = None
    _ONE_HOT: Final[bool] = True
    
    

    
    MAP_COLORS: Dict[int, str] = {
        0 : "#3c3c3c",
        1 : "#ff0000",
        2 : "#00ff00",
        3 : "#0000ff"
    }
    
    COLOR_LABEL_MAP : Dict[int, str] = {
        0 : "Unknow",
        1 : "Vineyards",
        2 : "Fruit trees",
        3 : "Olive groves"
    }
    

    
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
        batch_size: int = 1, 
        num_workers: int  = 1,
        useTemporalSize: bool = True,
        args: Namespace | None = None
    ):
        
        super().__init__(datasetFolder, batch_size, num_workers, args)
        
        
        self._TRAINING: PermanentCrops | None = None
        self._VALIDATION: PermanentCrops | None = None
        self._TEST: PermanentCrops | None = None
        self._setup_done = False
        self._useTemporalSize = useTemporalSize
        
        self._training_trasforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            
        ])
        
        self._test_trasforms = transforms.Compose([
            
        ])
        
    @property 
    #@lru_cache
    def use_oneHot_encoding(self) -> bool:
        return PermanentCrops_DataModule._ONE_HOT
    
    # @property 
    # def getIgnoreIndexFromLoss(self) -> int:
    #     return 0
    
    @property
    def input_channels(self) -> int:
        return PermanentCrops_DataModule.ImageChannels
    
    @property
    def output_classes(self) -> int:
        return PermanentCrops_DataModule.ClassesCount
    
    @property    
    def input_size(self) -> list[int]:
        if self._useTemporalSize:
            return [1, PermanentCrops_DataModule.ImageChannels, PermanentCrops_DataModule.TemporalSize, PermanentCrops_DataModule.ImageHeight, PermanentCrops_DataModule.ImageWidth]
        else: 
            return [1, PermanentCrops_DataModule.ImageChannels, PermanentCrops_DataModule.ImageHeight, PermanentCrops_DataModule.ImageWidth]
        
    def setup(self, stage=None) -> None:
        if self._setup_done:
            return
        
        dataSizeParametre = {
            "width" :  PermanentCrops_DataModule.ImageWidth,
            "height" : PermanentCrops_DataModule.ImageHeight,
            "channels" : PermanentCrops_DataModule.ImageChannels,
            "temporalSize" : PermanentCrops_DataModule.TemporalSize,
            "classesCount" : PermanentCrops_DataModule.ClassesCount,
            "useOneHotEncoding" : PermanentCrops_DataModule._ONE_HOT,
            "useTemporalSize" : self._useTemporalSize
        }
        
        self._TRAINING   = PermanentCrops(args=self._args, dataSize = dataSizeParametre, folderPath = self._datasetFolder, mode= DatasetMode.TRAINING, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        self._VALIDATION = PermanentCrops(args=self._args, dataSize = dataSizeParametre, folderPath = self._datasetFolder, mode= DatasetMode.VALIDATION, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._TEST       = PermanentCrops(args=self._args, dataSize = dataSizeParametre, folderPath = self._datasetFolder, mode= DatasetMode.TEST, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._setup_done = True
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAINING, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        #return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VALIDATION, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def calculate_classes_weight(self) ->torch.tensor:
        if not self._setup_done:
            self.setup()
            
        filePath = os.path.join(Globals.TEMP_DATA,f"{self.__class__.__name__}", "classes_weight.pickle")
        
        if Globals.USE_CACHAING and os.path.exists(filePath):
            with open(filePath, "rb") as f:
                return pickle.load(f)
            
            
        core: int | None = min(os.cpu_count(), 14)
        batch_size = 2
        
        if core == None:
            core = 0 
        else:
            batch_size = core * 20
        
        self._TRAINING.setLoadOnlyY(True)
        
        temp_loader = DataLoader(
            self._TRAINING,  # Assuming self._TRAIN is a Dataset object
            batch_size=batch_size,  # Set batch_size=1 for individual sample processing
            num_workers=core,
            shuffle=False,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=None
        )
        
        Globals.APP_LOGGER.info("Calulating classes weight...")
        
        y_flat: torch.Tensor = torch.empty((len(self._TRAINING) * PermanentCrops.ImageWidth * PermanentCrops.ImageHeight), dtype=torch.uint8)
        index: int = 0
        #self._TRAINING.useNormalizedData(False)
        #maxVale: float = 0.0

        for i, (x, y) in enumerate(temp_loader):
            print(f"Loading: {min(i + 1, len(temp_loader))}/{len(temp_loader)}", end="\r")
            
            Xmax = x.max()
            
            # if Xmax > maxVale:
            #     maxVale = Xmax
            #     print(maxVale)
            
            y = y.flatten()
            elementCount = y.shape[0]
            y_flat[index: index + elementCount] = y
            index += elementCount
            
        print()
        #Globals.APP_LOGGER.info(f"Max x value on dataset: {maxVale}")
        #self._TRAINING.useNormalizedData(True)
    
        
        y_flat = y_flat.numpy()
        available_classes = np.unique(y_flat)
        
        class_weights = compute_class_weight(
            class_weight='balanced',  # Opzione per bilanciare in base alla frequenza
            classes=available_classes,  # Array di tutte le classi
            y=y_flat
        )
        
        # Mappa i pesi su tutte le classi, assegnando peso 0 alle classi assenti
        weights = np.zeros(PermanentCrops.ClassesCount, dtype=np.float32)
        weights[available_classes] = class_weights
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        Globals.APP_LOGGER.info(f"Calculated classes weight: {weights_tensor}")

        self._TRAINING.setLoadOnlyY(False)
        
        
        if Globals.USE_CACHAING:
            folder = os.path.dirname(filePath)
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            with open(filePath, "wb") as f:
                pickle.dump(weights_tensor, f)
        
        return weights_tensor
    
         
            
    def on_work(self, model: ModelBase, device: torch.device, **kwargs) -> None:
        
        
        model.to(device)
        model.eval()
        
        
        idx = kwargs["idx"]
        confMatrix: ConfusionMatrix  = ConfusionMatrix(classes_number = 4, ignore_class=self.classesToIgnore(), mapFuntion=self.map_classes)
        if idx >= 0:
            idx = idx % len(self._TEST)

            with torch.no_grad():  
                data: Dict[str, any] = self._TEST.getItem(idx)
                
                x = data['x']
                y = data['y']
                y = y.unsqueeze(0)            #aggiungo la dimensione dela batch
                x = x.to(device).unsqueeze(0) #aggiungo la dimensione dela batch
                
                #x = torch.rand([1, 13, 62, 96, 96]).to(device)
             
                y_hat = model(x)
                
                
                
                # # tensor = y_hat.cpu().detach().squeeze(0).numpy()
                # torch.set_printoptions(profile="full")
                # with open("tensore.txt", "w") as f:
                #     f.write(str(x))
                    
                # print("Tensore salvato in 'tensor.txt'")
            
                
                
                
                #torch.set_printoptions(profile="full")
                
                y_hat_ = torch.argmax(y_hat, dim=1)
                y_ = torch.argmax(y, dim=1)
                
    
                y_ = y_.cpu().detach()
                y_hat_ = y_hat_.cpu().detach()
                
                confMatrix.update(y_pr=y_hat_, y_tr=y_)
                _, graphData = confMatrix.compute(showGraph=False)
                
                confMatrix.reset()
            
                self.show_processed_sample(x, y_hat_, y_, idx, graphData)
            return
        
        
    def show_processed_sample(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        index: int,
        confusionMatrixData: Dict[str, any],
        X_as_Int: bool = False,
        temporalSequenze: bool = True
    ) -> None:
        assert x is not None, "x is None"
        assert y_hat is not None, "y_hat is None"
        assert y is not None, "y is None"

        if len(y_hat.shape) == 3:
            y_hat = y_hat.squeeze(0)
        if len(y.shape) == 3:
            y = y.squeeze(0)

        x = x.cpu().detach()
        x = x.squeeze(0)  # Remove batch dimension
        

        if temporalSequenze:
            x = x.permute(1, 0, 2, 3)  # Rearrange dimensions for temporal data
            x = x.reshape(-1, PermanentCrops_DataModule.ImageWidth, PermanentCrops_DataModule.ImageHeight)

        if X_as_Int:
            x = x.int()
        else:
            #x = ((x * 1e4) / (2**16 - 1))
            x *= 255
            x = x.int()
            #x += 40
            x = x.clamp(0, 255)

        y_hat = y_hat.cpu().detach().int()
        y = y.cpu().detach()

        
        rowElement: int = 8
        num_images: int = int(x.shape[0] // 13)  # Number of images in batch
        num_cols: int = (num_images // rowElement) + (num_images % rowElement != 0)

        # Additional Windows for Other Bands
         # Band groups for visualization
        band_groups = {
            "Natural Color": [BandeIndex.BAND_4_BLUE.value, BandeIndex.BAND_3_GREEN.value, BandeIndex.BAND_2_RED.value],
            "Color Infrared": [BandeIndex.BAND_8_NIR.value, BandeIndex.BAND_4_BLUE.value, BandeIndex.BAND_3_GREEN.value],
            "Short-Wave Infrared": [BandeIndex.BAND_12_SWIR2.value, BandeIndex.BAND_A8.value, BandeIndex.BAND_4_BLUE.value],
            "Agriculture": [BandeIndex.BAND_11_SWIR1.value, BandeIndex.BAND_8_NIR.value, BandeIndex.BAND_2_RED.value],
            "Geology": [BandeIndex.BAND_12_SWIR2.value, BandeIndex.BAND_11_SWIR1.value, BandeIndex.BAND_2_RED.value],
            "Bathymetric": [BandeIndex.BAND_4_BLUE.value, BandeIndex.BAND_3_GREEN.value, BandeIndex.BAND_1_AEROSOL.value],
        }

        # Plot all band combinations
        for group_name, band_indices in band_groups.items():
            fig_group, axes_group = plt.subplots(rowElement, num_cols, figsize=(10, 12))
            fig_group.suptitle(f"Index {index}: {group_name} Bands")
            
            fig_group.tight_layout(pad=1.5)
            #fig_group.subplots_adjust(right=0.8)
            
            for idx in range(num_images):
                row: int = idx % rowElement
                col: int = idx // rowElement

                multiChannelImage = x[idx * 13:(idx + 1) * 13, :, :]
                band_images = [multiChannelImage[b] for b in band_indices]

                composite_image = torch.stack(band_images, dim=0).permute(1, 2, 0).numpy()

                ax = axes_group[row, col]
                ax.imshow(composite_image, interpolation='none')
                ax.set_title(f"Image {idx+1}")
                ax.axis('off')
                
        fig_group, axes_group = plt.subplots(rowElement, num_cols, figsize=(10, 12))
        fig_group.suptitle(f"Index {index}: Vegetation Index")
        
        fig_group.tight_layout(pad=1.5)
        #fig_group.subplots_adjust(right=0.8)
        
        for idx in range(num_images):
            row: int = idx % rowElement
            col: int = idx // rowElement

            multiChannelImage = x[idx * 13:(idx + 1) * 13, :, :]
            composite_image = multiChannelImage.permute(1, 2, 0).numpy()

            vegetation_index = (composite_image[:, :, BandeIndex.BAND_8_NIR.value] - composite_image[:, :, BandeIndex.BAND_4_BLUE.value]) / (composite_image[:, :, BandeIndex.BAND_8_NIR.value] + composite_image[:, :, BandeIndex.BAND_4_BLUE.value])
            

            ax = axes_group[row, col]
            ax.imshow(vegetation_index, interpolation='none', cmap='nipy_spectral')
            ax.set_title(f"Image {idx+1}")
            ax.axis('off')
            
        fig_group, axes_group = plt.subplots(rowElement, num_cols, figsize=(10, 12))
        fig_group.suptitle(f"Index {index}: Moisture Index")
        
        fig_group.tight_layout(pad=1.5)
        #fig_group.subplots_adjust(right=0.8)
        
        for idx in range(num_images):
            row: int = idx % rowElement
            col: int = idx // rowElement

            multiChannelImage = x[idx * 13:(idx + 1) * 13, :, :]
            composite_image = multiChannelImage.permute(1, 2, 0).numpy()

            moisture_index = (composite_image[:, :, BandeIndex.BAND_A8.value] - composite_image[:, :, BandeIndex.BAND_11_SWIR1.value]) / (composite_image[:, :, BandeIndex.BAND_A8.value] + composite_image[:, :, BandeIndex.BAND_11_SWIR1.value])
            

            ax = axes_group[row, col]
            ax.imshow(moisture_index, interpolation='none', cmap='plasma')
            ax.set_title(f"Image {idx+1}")
            ax.axis('off')
        

        # # Vegetation Index (B8-B4)/(B8+B4)
        # vegetation_index = (x[:, BandeIndex.BAND_8_NIR.value] - x[:, BandeIndex.BAND_4_BLUE.value]) / (x[:, BandeIndex.BAND_8_NIR.value] + x[:, BandeIndex.BAND_4_BLUE.value])
        

        # # Moisture Index (B8A-B11)/(B8A+B11)
        # moisture_index = (x[:, BandeIndex.BAND_A8.value] - x[:, BandeIndex.BAND_11_SWIR1.value]) / (x[:, BandeIndex.BAND_A8.value] + x[:, BandeIndex.BAND_11_SWIR1.value])
        
        # print(vegetation_index.shape)

        # # Show Vegetation and Moisture Indices for each image in the sequence
        # for idx in range(num_images):
        #     # Vegetation Index Visualization per Image
        #     fig_veg, ax_veg = plt.subplots(figsize=(5, 5))
        #     ax_veg.imshow(vegetation_index[idx].numpy(), cmap='YlGn', vmin=-1, vmax=1)
        #     ax_veg.set_title(f"Vegetation Index - Image {idx+1}")
        #     ax_veg.axis('off')

        #     # Moisture Index Visualization per Image
        #     fig_moist, ax_moist = plt.subplots(figsize=(5, 5))
        #     ax_moist.imshow(moisture_index[idx].numpy(), cmap='Blues', vmin=-1, vmax=1)
        #     ax_moist.set_title(f"Moisture Index - Image {idx+1}")
        #     ax_moist.axis('off')

        # Label and Prediction Maps
        color_list = [color for _, color in sorted(PermanentCrops_DataModule.MAP_COLORS.items())]
        cmap = ListedColormap(color_list)

        label_map = y.numpy()
        pred_map = y_hat.numpy()

        fig_map, axes_map = plt.subplots(1, 2, figsize=(10, 5))
        fig_map.suptitle("Feature Maps: Label Map and Prediction Map")

        axes_map[0].imshow(label_map, interpolation='none', cmap=cmap, vmin=0, vmax=3)
        axes_map[0].set_title("Label Map")
        axes_map[0].axis('off')

        axes_map[1].imshow(pred_map, interpolation='none', cmap=cmap, vmin=0, vmax=3)
        axes_map[1].set_title("Prediction Map")
        axes_map[1].axis('off')

        legend_patches = [
            mpatches.Patch(color=PermanentCrops_DataModule.MAP_COLORS[cls], label=f'{cls} - {label}')
            for cls, label in PermanentCrops_DataModule.COLOR_LABEL_MAP.items()
        ]
        fig_map.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc='upper right', title="Class Colors")
        fig_map.subplots_adjust(right=0.7)

        # Confusion Matrix
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

        # Save all figures
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            fig.savefig(os.path.join(Globals.MATPLOTLIB_OUTPUT_FOLDER, f"grafico_{fignum}.png"))

        

    #plt.show()