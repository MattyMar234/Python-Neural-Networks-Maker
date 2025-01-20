from argparse import Namespace
from ast import Tuple
from functools import lru_cache
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
import cv2

from sklearn.utils.class_weight import compute_class_weight
from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import TableBase, TensorTable
from DatasetComponents.Datasets.DatasetBase import PostgresDataset_Interface
from DatasetComponents.Datasets.munich480 import Munich480
import Globals
from Networks.Metrics.ConfusionMatrix import ConfusionMatrix
from Networks.NetworkComponents.NeuralNetworkBase import *
from Utility.TIF_creator import TIF_Creator

from .DataModuleBase import *
from torch.utils.data import DataLoader
import os
import time

class BandeIndex(Enum):
    BAND_1_AEROSOL: int = 10
    BAND_2_BLUE: int = 2
    BAND_3_GREEN: int = 1
    BAND_4_RED: int = 0
    BAND_5_VRE1: int = 4
    BAND_6_VRE2: int = 5
    BAND_7_VRE3: int = 6
    BAND_8_NIR: int = 3
    BAND_A8: int = 7
    BAND_9_WV: int = 11
    BAND_10_SWIR: int = 12
    BAND_11_SWIR1: int = 8
    BAND_12_SWIR2: int = 9

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
        
    colors[0] = '#333333'
    
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
    _ONE_HOT: Final[bool] = True

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
        useTemporalSize: bool = False,
        args: Namespace | None = None
    ):
        super().__init__(datasetFolder, batch_size, num_workers, args)

        
        assert type(year) == Munich480.Year, f"year deve essere di tipo Munich480.Year"
        
        
        self._download = download
        self._TRAIN: Munich480 | None = None
        self._VAL: Munich480 | None = None
        self._TEST: Munich480 | None = None
        self._setup_done = False
        
        self._year = year
        
        self._useTemporalSize = useTemporalSize
        self._total_channel = 13
        self._classesMapping: dict = {}
        self._read_classes()
        
       
        
            
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
    #@lru_cache
    def use_oneHot_encoding(self) -> bool:
        return Munich480_DataModule._ONE_HOT
    
    @property 
    def getIgnoreIndexFromLoss(self) -> int:
        if vars(self._args)[Globals.USE_IGNORE_CLASS]:
            return 0
        else:
            return -100
    
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
            
        filePath = os.path.join(self._datasetFolder, "classes_weight.pt")
        
        if os.path.exists(filePath):
            with open(filePath, "rb") as f:
                return pickle.load(f)
            
            
        core: int | None = os.cpu_count()
        batch_size = 4
        
        if core == None:
            core = 0 
        else:
            batch_size = core * 20
        
        self._TRAIN.setLoadOnlyY(True)
        
        temp_loader = DataLoader(
            self._TRAIN,  # Assuming self._TRAIN is a Dataset object
            batch_size=batch_size,  # Set batch_size=1 for individual sample processing
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
        
        with open(filePath, "wb") as f:
            pickle.dump(weights_tensor, f)
        
        return weights_tensor


    def setup(self, stage=None) -> None:
        if self._setup_done:
            return
        
        self._TRAIN = Munich480(args=self._args, folderPath = self._datasetFolder, mode= Munich480.DatasetMode.TRAINING, year= self._year, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        self._VAL   = Munich480(args=self._args, folderPath = self._datasetFolder, mode= Munich480.DatasetMode.VALIDATION, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._TEST  = Munich480(args=self._args, folderPath = self._datasetFolder, mode= Munich480.DatasetMode.TEST, year= self._year, transforms=self._test_trasforms, useTemporalSize=self._useTemporalSize)
        self._setup_done = True


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._TRAIN, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
        #return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._VAL, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._TEST, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, pin_memory=self._pin_memory, persistent_workers=self._persistent_workers, drop_last=True, prefetch_factor=self._prefetch_factor)
    
    
    def on_work(self, model: ModelBase, device: torch.device, **kwargs) -> None:
        self.setup()
        
        # self._TRAIN1 = Munich480(args=self._args, folderPath = self._datasetFolder, mode= Munich480.DatasetMode.TRAINING, year= Munich480.Year.Y2016, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        # self._TRAIN2 = Munich480(args=self._args, folderPath = self._datasetFolder, mode= Munich480.DatasetMode.TRAINING, year= Munich480.Year.Y2016, transforms=self._training_trasforms, useTemporalSize=self._useTemporalSize)
        
        
        # self._TRAIN1.setLoadOnlyY(True)
        # self._TRAIN2.setLoadOnlyY(True)
        
        # self._TRAIN3 = torch.utils.data.ConcatDataset([self._TRAIN1, self._TRAIN2])
        
        # train_loader = DataLoader(
        #     self._TRAIN, 
        #     batch_size=self._batch_size, 
        #     num_workers=self._num_workers, 
        #     shuffle=True, 
        #     pin_memory=self._pin_memory, 
        #     persistent_workers=self._persistent_workers, 
        #     drop_last=True, 
        #     prefetch_factor=self._prefetch_factor
        # )
        
        # num_classes = 27
        # label_counts = np.zeros(num_classes, dtype=int)


        # print("lading data")
        # # Itera su tutti gli elementi del dataset
        
        # class_counts: torch.Tensor = torch.zeros(self.output_classes)
        
        # for idx, (_, y) in enumerate(train_loader):
        #     print(f"Loading: {min(idx + 1, len(train_loader))}/{len(train_loader)}", end="\r")
            
        #     #print(y.shape)
        #     y = torch.argmax(y, dim=1)
        #     # print(y.shape)
        #     # print(y)
        #     y_flat = y.flatten()
        #     class_counts += torch.bincount(y_flat, minlength=self.output_classes)
            
            
    

        # availableClass = []
        # value = []
        
        # for i in range(num_classes):
        #     if i in self._classesMapping:
        #         availableClass.append(i)
        #         value.append(class_counts[i])

        # # Grafico a barre per la distribuzione delle etichette
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(availableClass)), value, color='orange')
        # plt.xlabel('Crop classes')
        # plt.ylabel('Labels count')
        # #plt.title('Label Distribution per Class')
        # plt.xticks(range(len(availableClass)), labels=[f'{self.map_classes(i)}' for i in availableClass], rotation=45)
        # plt.grid(axis='y')
        # plt.tight_layout()
        # plt.show()
        
        # return
        
        # table: TableBase = TensorTable("munich")
        
        # databaseParametre = DatabaseParametre(
        #     host="192.168.1.202",#"host.docker.internal",
        #     port="44500",
        #     database="NAS1",
        #     user="postgres",
        #     password="admin",
        #     maxconn  = 10,
        #     timeout  = 10
        # )
        
        
        # remoteConnection: PostgresDataset_Interface = PostgresDataset_Interface(databaseParametre)
        # DB: PostgresDB = remoteConnection.getStream()
        # DB.execute_query(table.createTable_Query())
        
        # idx = 0
        
        # data: Dict[str, any] = self._TEST.getItems(idx)  
        # x = data['x']
        # y = data['y']  
        # profile = data['profile']  
        
        
        # tensor_binary_x = Binary(pickle.dumps(x))     
        # tensor_binary_y = Binary(pickle.dumps(y))
        # profile_binary = Binary(pickle.dumps(profile))
        # #DB.execute_query(table.insertElement_Query(id = idx, x = tensor_binary_x, y = tensor_binary_y, info = profile_binary))
        # q = table.getElementAt_Query(0)
        # print(q)
        
        idx = kwargs["idx"]

        if idx >= 0:
            #self.noIgnore(model, device, **kwargs)
            self.with_ignore(model, device, **kwargs)
        
        if idx == -1:
            
            temp_loader1 = DataLoader(
                self._TEST,  # Assuming self._TRAIN is a Dataset object
                batch_size=1,  # Set batch_size=1 for individual sample processing
                num_workers=kwargs["workers"],
                shuffle=False,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=None
            )
            
            temp_loader2 = DataLoader(
                self._VAL,  # Assuming self._TRAIN is a Dataset object
                batch_size=1,  # Set batch_size=1 for individual sample processing
                num_workers=kwargs["workers"],
                shuffle=False,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=None
            )
            
            temp_loader3 = DataLoader(
                self._TRAIN,  # Assuming self._TRAIN is a Dataset object
                batch_size=1,  # Set batch_size=1 for individual sample processing
                num_workers=kwargs["workers"],
                shuffle=False,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=None
            )
            
            creator_x:TIF_Creator = TIF_Creator('/app/geoData/x')
            creator_x_v2:TIF_Creator = TIF_Creator('/app/geoData/x_v2')
            creator_y:TIF_Creator = TIF_Creator('/app/geoData/y')
            creator_y_hat:TIF_Creator = TIF_Creator('/app/geoData/y_hat')
            #loaders = [temp_loader1, temp_loader2, temp_loader3]
            loaders = [temp_loader1]
            
            number = 240
            
            for i in range(len(temp_loader1)):
                print(f"Processing {i}/{number}", end = '\r')
                
                try:
                    output = temp_loader1.dataset.get_dataset_image(i,"20160212")
                    image = output["data"]
                    profile = output["profile"]
                    
                    red = image[BandeIndex.BAND_4_RED.value, :, :]
                    green = image[BandeIndex.BAND_3_GREEN.value, :, :]
                    blue = image[BandeIndex.BAND_2_BLUE.value, :, :]

                    rgb_image = np.stack([red, green, blue])
                    rgb_image = ((rgb_image * (2**16 - 1)))
                    #rgb_image = rgb_image.permute(1, 2, 0)
                    rgb_image_np = self.adjust_RGB_image(rgb_image)
                    
                    creator_x_v2.makeTIF(f'{i}.tif', profile, data =rgb_image_np, channels = 3, width=48, height=48)
                except Exception as e:
                    print(e)
                    pass 
            print("start merging x_v2...")
            creator_x_v2.mergeTIFs('/app/merged_x_v2.tif')
            return
            
            # creator_y_hat.mergeTIFs('/app/merged_y_hat.tif')
            # creator_y.mergeTIFs('/app/merged_y.tif')
            step = 0
            size = 0
            T = 3
            count = 140
            
            for loader in loaders:
                size += len(loader)
                
            for k, loader in enumerate(loaders):
                for n, (x, y) in enumerate(loader):
                    print(f"Processing {step}/{size} | [{k}-{n}]", end = '\r')
                    with torch.no_grad():
                        #profile = loader.dataset.getItemInfo(n)
                        #print(x.shape)
                        #image = x[0, :, T, :, :]
                        
                        output = loader.dataset.get_dataset_image(n,"20160212")
                        
                        image = output["data"]
                        profile = output["profile"]
                        
                        #print(image.shape)
                        
                        #print(image.shape)
                        red = image[BandeIndex.BAND_4_RED.value, :, :]
                        green = image[BandeIndex.BAND_3_GREEN.value, :, :]
                        blue = image[BandeIndex.BAND_2_BLUE.value, :, :]

                        #print(red.shape, green.shape, blue.shape)

                        rgb_image = np.stack([red, green, blue])
                        rgb_image = ((rgb_image * (2**16 - 1)))
                        #rgb_image = rgb_image.permute(1, 2, 0)
                        rgb_image_np = self.adjust_RGB_image(rgb_image)
                        
                        

                        # x = x.to(device)
                        # y_hat = model(x)

                        # y_hat_ = torch.argmax(y_hat, dim=1)
                        # y_ = torch.argmax(y, dim=1)


                        # #profile = self._TEST.getItemInfo(n)
                        # y_hat_RGB = np.zeros((3, 48, 48), dtype=np.uint8)
                        # y_RGB = np.zeros((3, 48, 48), dtype=np.uint8)
                        
                        # for i in range(48):
                        #     for j in range(48):
                        #         class_id = int(y_hat_[0, i, j])
                        #         y_hat_RGB[:, i, j] = self.MAP_COLORS_AS_RGB_LIST[class_id]

                        #         class_id = int(y_[0, i, j])
                        #         y_RGB[:, i, j] = self.MAP_COLORS_AS_RGB_LIST[class_id]

                    creator_x.makeTIF(f'{step}.tif', profile, data =rgb_image_np, channels = 3, width=48, height=48)
                    #creator_y_hat.makeTIF(f'{step}.tif', profile, data =y_hat_RGB, channels = 3, width=48, height=48) 
                    #creator_y.makeTIF(f'{step}.tif', profile, data =y_RGB, channels = 3, width=48, height=48)
                
                    step += 1
                    
                    if count >= 0 and step >= count:
                        break
        
            print("start merging x...")
            creator_x.mergeTIFs('/app/merged_x.tif')
            # print("start merging y_hat...")
            # creator_y_hat.mergeTIFs('/app/merged_y_hat.tif')
            # creator_y.mergeTIFs('/app/merged_y.tif')
   
    def noIgnore(self, model, device, **kwargs):
        confMatrix: ConfusionMatrix  = ConfusionMatrix(classes_number = 27, ignore_class=self.classesToIgnore(), mapFuntion=self.map_classes)
    
    
        checkpoint = torch.load(kwargs["ckpt_path"], map_location=torch.device(device))
        #print(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)
        model.eval()
        
        idx = kwargs["idx"]
        idx = idx % len(self._TEST)

        with torch.no_grad():  
            data: Dict[str, any] = self._TEST.getItem(idx)
            
            x = data['x']
            y = data['y']
            y = y.unsqueeze(0)
            x = x.to(device)
            
            y_hat = model(x.unsqueeze(0))
            
            y_hat_ = torch.argmax(y_hat, dim=1)
            y_ = torch.argmax(y, dim=1)
            
            print(y_hat_.shape, y_.shape)

            y_ = y_.cpu().detach()
            y_hat_ = y_hat_.cpu().detach()
            
            
            confMatrix.update(y_pr=y_hat_, y_tr=y_)
            _, graphData = confMatrix.compute(showGraph=False)
            
            confMatrix.reset()
        
            self.show_processed_sample(x, y_hat_, y_, idx, graphData)
            
    def with_ignore(self, model, device, **kwargs):
        
        confMatrix: ConfusionMatrix  = ConfusionMatrix(classes_number = 27, ignore_class=self.classesToIgnore(), mapFuntion=self.map_classes)
    
    
        checkpoint = torch.load(kwargs["ckpt_path"], map_location=torch.device(device))
        #print(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)
        model.eval()
        
        idx = kwargs["idx"]
        idx = idx % len(self._TEST)

        with torch.no_grad():  
            data: Dict[str, any] = self._TEST.getItem(idx)
            
            x = data['x']
            y = data['y']
            y = y.unsqueeze(0)
            x = x.to(device)
            
            y_hat = model(x.unsqueeze(0))
            
            # y_hat_ = torch.argmax(y_hat, dim=1)
            # y_ = torch.argmax(y, dim=1)
            
            # print(y_hat_.shape, y_.shape)

            # y_ = y_.cpu().detach()
            # y_hat_ = y_hat_.cpu().detach()
            
            
                # Calcola la probabilità di ogni classe
            y_hat_probs = torch.softmax(y_hat, dim=1)  # Applicare softmax per ottenere probabilità
            max_probs, y_hat_ = torch.max(y_hat_probs, dim=1)  # Trova la classe con probabilità massima

            # Calcola la classe di verità
            y_ = torch.argmax(y, dim=1)
            
            # Stampa le forme
            print(y_hat_.shape, y_.shape)
            
            # Verifica la confidenza
            confidence_threshold = 0.75  # Soglia di confidenza per la classe "sconosciuto"
            y_hat_ = torch.where(max_probs < confidence_threshold, torch.zeros_like(y_hat_), y_hat_)
            
            y_hat_ = y_hat_.cpu().detach()
            y_ = y_.cpu().detach()
            
            confMatrix.update(y_pr=y_hat_, y_tr=y_)
            _, graphData = confMatrix.compute(showGraph=False)
            
            confMatrix.reset()
        
            self.show_processed_sample(x, y_hat_, y_, idx, graphData)
        
    
    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    
    
    def adjust_RGB_image(self, rgb_image: np.ndarray | torch.Tensor) -> np.array:
        
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.numpy()
            
        rgb_image /= 10000
        rgb_image *=255
        #rgb_image = rgb_image.clamp(0.0, 255.0)
        #rgb_image = rgb_image.int()
        rgb_image = rgb_image.astype(np.uint8)
        rgb_image = self.adjust_gamma(rgb_image, 2.20)
        
        return rgb_image
    
    def show_processed_sample(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, index: int, confusionMatrixData: Dict[str, any], X_as_Int: bool = False, temporalSequenze = True) -> None:
        assert x is not None, "x is None"
        assert y_hat is not None, "y_hat is None"
        assert y is not None, "y is None"
        
        
        if len(y_hat.shape) == 3:
            y_hat = y_hat.squeeze(0)
        if len(y.shape) == 3:
            y = y.squeeze(0)
        
        print(y_hat.shape, y.shape)
        
        x = x.cpu().detach()
        x = x.squeeze(0) # elimino la dimensione della batch
        
        
        if temporalSequenze:
            x = x.permute(1, 0, 2, 3)
            x = x.reshape(-1, 48, 48)
            
            
            #x = ((x * (2**16 - 1))/10000)
            # x *= 255
            # #x += 30
            # x = x.clamp(0.0, 255.0)
            # x = x.int()
            
        x = ((x * (2**16 - 1)))
        
        y_hat = y_hat.cpu().detach()
        y = y.cpu().detach()
        
        
        #fig, axes = plt.subplots(rowElement, (num_images // rowElement) + (num_images % rowElement != 0) + 3, figsize=(16, 12))  # Griglia verticale per ogni immagine

        
        # band_combinations = {
        #     "True Color (RGB)": [2, 1, 0],
        #     "False Color (NIR)": [3, 2, 1],
        #     "False Color Urban": [11, 3, 2],
        #     "Agriculture": [11, 8, 2],
        #     "Atmospheric Penetration": [11, 12, 8],
        #     "Vegetation Analysis": [7, 3, 2],
        #     "Moisture Index": [8, 11, 12],
        #     "Natural Color (SWIR)": [3, 2, 11],
        #     "Geology": [12, 11, 2],
        # }
        
        band_combinations = {
            "True Color (RGB)": [
                BandeIndex.BAND_4_RED.value,
                BandeIndex.BAND_3_GREEN.value, 
                BandeIndex.BAND_2_BLUE.value
            ],
            "Color Infrared" : [
                BandeIndex.BAND_8_NIR.value,
                BandeIndex.BAND_4_RED.value,
                BandeIndex.BAND_3_GREEN.value
            ],
            "Short-Wave Infrared" : [
                BandeIndex.BAND_12_SWIR2.value,
                BandeIndex.BAND_A8.value,
                BandeIndex.BAND_4_RED.value
            ],
            "a" : [3], 
            "b" : [4],
            "c" : [5], 
            "d" : [6],
            "e" : [7], 
            "f" : [8],
            "g" : [9], 
            "h" : [10],
            "i" : [11], 
            "j" : [12],
        }

        # Define plot layout
        num_rows = len(band_combinations)
        num_cols = 32  # One column per image in the sequence
        fig1, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.2, num_rows * 2))
        #fig1.suptitle(f"Index {index}: Band Compositions", fontsize=16)
        fig1.subplots_adjust(wspace=0.1, hspace=0.1)  # Riduci lo spazio tra immagini

        for row_idx, (name, bands) in enumerate(band_combinations.items()):
            for col_idx in range(num_cols):
                # Extract image for current sequence and select bands
                image = x[col_idx * 13:(col_idx + 1) * 13, :, :]
                
                if len(bands) == 3:
                
                    red, green, blue = image[bands[0]], image[bands[1]], image[bands[2]]
                    rgb_image = torch.stack([red, green, blue], dim=0).permute(1, 2, 0)

                    rgb_image = self.adjust_RGB_image(rgb_image)
                
                    #x = percentile_stretch(x, lower_percent=2, upper_percent=98)*255
                    #x = torch.tensor(x)

                    # Plot on respective axes
                    ax = axes[row_idx, col_idx]
                    ax.imshow(rgb_image.astype('uint8'), interpolation='None')
                    
                    if col_idx == 0:
                        ax.set_ylabel(name, fontsize=10)
                    ax.axis('off')
                    
                elif len(bands) == 1:
                    # Visualizza singola banda
                    single_band = image[bands[0]].numpy()  # Accedi alla banda specifica
                    
                    single_band /= 10000
                    single_band *= (2**16 - 1)
                    
                    ax = axes[row_idx, col_idx]
                    ax.imshow(single_band.astype('uint16'), cmap='magma', interpolation='none')
                    
                    if col_idx == 0:
                        ax.set_ylabel(name, fontsize=10)
                    ax.axis('off')
                                    
                if row_idx == 0:
                    ax.set_title(f"t{col_idx+1}", fontsize=8)

        # Add label for the row (band combination name)
        if col_idx == 0:
            ax.set_ylabel(name, fontsize=10)
        
        ax.text(-10, image.shape[1] // 2, name, va='center', ha='right', fontsize=10, rotation=90)
        

        #=========================================================================#
        # # Prima finestra: Visualizzazione delle immagini RGB
        # rowElement: int = 8
        # num_images: int = int(x.shape[0] // 13)  # Numero di immagini nel batch
        # num_cols: int = (num_images // rowElement) + (num_images % rowElement != 0)
        
        # fig1, axes1 = plt.subplots(rowElement, num_cols, figsize=(10, 12))
        # fig1.suptitle(f"Index {index}")
    
        
        # for idx in range(num_images):
        #     row: int = idx % rowElement
        #     col: int = idx // rowElement
            
        #     # Estrazione dell'immagine (13 canali)
        #     image = x[idx * 13:(idx + 1) * 13, :, :]
        #     red, green, blue = image[2], image[1], image[0]
        #     rgb_image = torch.stack([red, green, blue], dim=0).permute(1, 2, 0).numpy()

        #     ax = axes1[row, col]
        #     ax.imshow(rgb_image)
        #     ax.set_title(f"Image {idx+1}")
            #ax.axis('off')
        
        
        # Crea una colormap personalizzata
        color_list = [color for _, color in sorted(Munich480_DataModule.MAP_COLORS.items())]
        cmap = ListedColormap(color_list)
        
        label_map = y.numpy()         # Etichetta per l'immagine corrente
        pred_map = y_hat.numpy()      # Predizione con massimo di ciascun layer di `y_hat`
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle("Label Map and Prediction Map")
        
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
        
        #fig1.tight_layout(pad=2.0)
        plt.tight_layout(pad=2.0, rect=[0, 0, 0, 0])
        
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
        