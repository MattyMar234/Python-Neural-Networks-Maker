import math
import random
from matplotlib import pyplot as plt
import matplotlib
import rasterio


from .ImageDataset import *
from .DatasetBase import *
from PIL import Image

import torch
import torch.nn.functional as F

import os
from os import listdir
from enum import Enum, auto, Flag
from typing import Final, Tuple
import asyncio
import aiofiles

from threading import Lock
from functools import lru_cache
from functools import lru_cache

class Munich480(Segmentation_Dataset_Base):
    
    _EVAL_TILEIDS_FOLDER = os.path.join("tileids","val_folders.txt")
    _TRAIN_TILEIDS_FOLDER = os.path.join("tileids", "train_folders.txt")
    _TEST_TILEIDS_FOLDER = os.path.join("tileids", "test_folders.txt")
    
    _semaphore = asyncio.Semaphore(10)
    _classLock = Lock()
    _stack_axis: int = 1
    
    _dataInitialized: bool = False
    
    TemporalSize: Final[int] = 32
    ImageChannels: Final[np.array] = np.array([4,6,3])
    ImageChannelsCount: Final[int] = np.sum(ImageChannels)
    ImageWidth: Final[int] = 48
    ImageHeight: Final[int] = 48
    
    
    
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
        
    
    
    
    
    
    
    
    @lru_cache(maxsize=27)
    @staticmethod
    def mapClassValue(index: int) -> str:
        assert index < 27 and index >= 0, "Index out of range"
        assert Munich480._dataInitialized, "Classes not initialized"
        return Munich480._classesMapping[index]
    



    def __new__(cls, folder, *args, **kwargs):
        #cls._init_shared_data(datasetFolder=folder)
        return super().__new__(cls)
           
  
    def __init__(self, folderPath:str | None, mode: DataType, year: Year, distance: Distance, transforms, useTemporalSize: bool = False):
        
        assert type(mode) == Munich480.DataType, "Invalid mode type"
        assert type(year) == Munich480.Year, "Invalid year type"
        assert type(distance) == Munich480.Distance, "Invalid distance type"
        
        total_channel = 0
        
        if Munich480.Distance.m10 in distance:
            total_channel += Munich480.ImageChannels[0]
        if Munich480.Distance.m20 in distance:
            total_channel += Munich480.ImageChannels[1]
        if Munich480.Distance.m60 in distance:
            total_channel += Munich480.ImageChannels[2]
        
        self._total_channels = total_channel#Munich480ImageChannels * Munich480._TemporalSize * temp_count
            
        if not useTemporalSize:
            total_channel *= Munich480.TemporalSize
        
        
        Segmentation_Dataset_Base.__init__(
            self, 
            imageSize = (Munich480.ImageWidth, Munich480.ImageHeight, total_channel, Munich480.TemporalSize if not useTemporalSize else 0), 
            classesCount = 27, 
            x_transform=transforms,
            y_transform = transforms,
            oneHot = True
        )
        
        
        self._folderPath:str | None = folderPath
        self._years: Munich480.Year = year
        self._distance: Munich480.Distance = distance
        self._useTemporalSize: bool = useTemporalSize
        
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
    
    def get_dates(self, path: str, sample_number= int | None) -> list[str]:
        #assert os.path.exists(path), f"Path {path} does not exist"
    
        files = os.listdir(path)
        dates = list()
        
        for f in files:
            date = f.split("_")[0]
            if len(date) == 8:  # 20160101
                dates.append(date)

        dates = list(set(dates))
        
        if sample_number is None or sample_number < 0:
            return dates
        
        if len(dates) > sample_number:
            dates = random.sample(dates, sample_number)
        
        elif len(dates) < sample_number:
            while len(dates) < sample_number:
                dates.append(random.choice(dates))
        
        dates.sort()
        return dates
    
    
    def _normalize_dif_data(self, data: np.array) -> np.array:
        # """Normalizza i dati tra 0 e 255."""
        # return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        return data.astype(np.float32) * 1e-4
    
    def _load_dif_file(self, filePath:str, normalize: bool = True) -> np.array:
        #async with Munich480._semaphore:
        with rasterio.open(filePath) as src:
            data = src.read()
            
            if normalize:
                return self._normalize_dif_data(data)
            return data
                
        
    def _load_y(self, folder:str):
        return self._load_dif_file(filePath=os.path.join(folder, "y.tif"), normalize = False)
        
        
    def _load_year_sequenze(self, year: str, number: str):
        sequenzeFolder = os.path.join(self._folderPath, year, str(number))
        dates = self.get_dates(path=sequenzeFolder, sample_number=Munich480.TemporalSize)

        #torch.empty
        x: torch.Tensor = torch.empty((Munich480.TemporalSize, self._total_channels, Munich480.ImageHeight, Munich480.ImageWidth), dtype=torch.float32)
        
        # Mappa per associare le distanze ai suffissi dei file
        distance_map = {
            Munich480.Distance.m10: "_10m.tif",
            Munich480.Distance.m20: "_20m.tif",
            Munich480.Distance.m60: "_60m.tif"
        }

        # Itera attraverso ogni data per caricare i dati temporali
        for t, date in enumerate(dates):
            current_channel_index = 0  # Reset per ogni t-step

            # Carica i dati per ciascuna distanza selezionata
            for distance, suffix in distance_map.items():
                if distance in self._distance:
                    data = self._load_dif_file(os.path.join(sequenzeFolder, f"{date}{suffix}"))
                    tensor = torch.from_numpy(data).unsqueeze(0)  # Aggiungi dimensione per l'interpolazione
                    tensor = F.interpolate(tensor, size=(Munich480.ImageHeight, Munich480.ImageWidth))
                    num_channels = tensor.size(1)
                    
                    # Assegna il tensor ai canali specifici per il timestamp t
                    x[t, current_channel_index:current_channel_index + num_channels, :, :] = tensor.squeeze(0)
                    current_channel_index += num_channels  # Aggiorna l'indice dei canali

        # Carica la maschera di segmentazione
        y = self._load_y(sequenzeFolder)
        return x, y
    
    # def _load_year_sequenze(self, year: str, number: str):
    #     sequenzeFolder = os.path.join(self._folderPath, year, str(number))
    #     dates = self.get_dates(path=sequenzeFolder, sample_number=Munich480.TemporalSize)

    #     #torch.empty
    #     x: torch.Tensor = torch.empty((Munich480.TemporalSize, Munich480.ImageChannels.sum(), Munich480.ImageHeight, Munich480.ImageWidth), dtype=torch.float32)
    #     current_channel_index: int = 0

    #     #print(x.shape)

    #     for t, date in enumerate(dates):
    #         current_channel_index = 0
            
    #         #print(f"temporale instance {t}: {date}")

    #         if Munich480.Distance.m10 in self._distance:
    #             data = self._load_dif_file(os.path.join(sequenzeFolder, f"{date}_10m.tif"))
    #             tensor = torch.from_numpy(data) 
    #             tensor = F.interpolate(tensor.unsqueeze(0), size=(Munich480.ImageHeight, Munich480.ImageWidth))
    #             num_channels = tensor.size(1)
                
    #             # torch.set_printoptions(profile="full")
    #             # torch.set_printoptions(linewidth=200)
    #             # print("prima: ", x[0, 0:5])
                
    #             # print(f"x10 aded in: [{t},{current_channel_index}:{current_channel_index + num_channels}]")
    #             x[t, current_channel_index:current_channel_index + num_channels, :, :] = tensor  # Assegna i canali
    #             current_channel_index += num_channels  
                
    #             # print("dopo: ", x[0, 0:5])
    #             # os._exit(0)

    #         if Munich480.Distance.m20 in self._distance:
    #             data = self._load_dif_file(os.path.join(sequenzeFolder, f"{date}_20m.tif"))
    #             tensor = torch.from_numpy(data)  
    #             tensor = F.interpolate(tensor.unsqueeze(0), size=(Munich480.ImageHeight, Munich480.ImageWidth))
    #             num_channels = tensor.size(1)  
                
    #             #print(f"x20 aded in: [{t},{current_channel_index}:{current_channel_index + num_channels}]")
    #             x[t, current_channel_index:current_channel_index + num_channels, :, :] = tensor  # Assegna i canali
    #             current_channel_index += num_channels  
                

    #         if Munich480.Distance.m60 in self._distance:
    #             data = self._load_dif_file(os.path.join(sequenzeFolder, f"{date}_60m.tif"))
    #             tensor = torch.from_numpy(data)  
    #             tensor = F.interpolate(tensor.unsqueeze(0), size=(Munich480.ImageHeight, Munich480.ImageWidth))
    #             num_channels = tensor.size(1)  
                
    #             #print(f"x60 aded in: [{t},{current_channel_index}:{current_channel_index + num_channels}]")
    #             x[t, current_channel_index:current_channel_index + num_channels, :, :] = tensor  # Assegna i canali
    #             current_channel_index += num_channels  
                        
    #     y = self._load_y(sequenzeFolder)
    #     return x, y
    
    
        
                
    def _getSize(self) -> int:
        return np.size(self._dataSequenze)        

    def _load_data(self, idx: int):
        
        if Munich480.Year.Y2016 in self._years and not (Munich480.Year.Y2017 in self._years):
            return self._load_year_sequenze("data16", str(idx))
        
        elif not (Munich480.Year.Y2016 in self._years) and Munich480.Year.Y2017 in self._years:
            return self._load_year_sequenze("data17", str(idx))
        
        else:
        
            x16: torch.Tensor | None = None
            x17: torch.Tensor | None = None
            y16: torch.Tensor | None = None
            y17: torch.Tensor | None = None
        
       
            x16, y16 = self._load_year_sequenze("data16", str(idx))
            x17, _ = self._load_year_sequenze("data17", str(idx))
            
            return torch.cat((x16, x17)), y16

    @DatasetBase._FormatResult
    def _getItem(self, idx: int) -> any:
        idx = self._mapIndex(idx)
    
        #x, y = asyncio.run(self._load_data(idx))
        x, y = self._load_data(idx)
        
        if not self._useTemporalSize:
            x = x.view(-1, Munich480.ImageHeight, Munich480.ImageWidth)
            x = x.permute(1, 2, 0)
            # x = np.transpose(x, (1, 2, 0))
        else:
            # permute channels with time_series (t x c x h x w) -> (c x t x h x w)
            x = x.permute(1, 0, 2, 3)
        
        #y = torch.squeeze(y, dim=0)
        y = np.squeeze(y, axis=0)
        
        #x = x.float()
        #x = x / 255
    
        
        
        #print(x.shape, y.shape) ---> torch.Size([48, 48, 832]) np.size(48, 48)
        
        return x, y 
            
    
    def adjustData(self, sample: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        x, y = sample
        y = y.permute(2,0,1)
        y = y.float()
        
        # for i in range(y.shape[0]):
        #     print(f"Layer{i}:{y[i]}")
        
        print(x)
        
        return x, y
            
        
    def show_sample(self, sample) -> None:
        # Assumiamo che `sample` abbia la forma [C, H, W]
        
        # Creare una directory per salvare le immagini se non esiste già
        os.makedirs('/app/src/imgs', exist_ok=True)

        # Ottieni il numero di canali
        num_channels = sample.shape[0]

        # Inizializza una lista per contenere le immagini dei canali
        images = []

        # Iterare sui canali e convertire ognuno in un'immagine PIL
        for i in range(num_channels):
            # Normalizza il canale
            channel_data = sample[i].numpy()
            img = Image.fromarray((channel_data * 255).astype('uint8'))
            
            # Applicare una mappa di colori (utilizzando 'hot' come esempio)
            img_colored = plt.cm.viridis(np.array(img) / 255.0)[:, :, :3]  # Ignoriamo l'alpha
            img_colored = (img_colored * 255).astype(np.uint8)
            images.append(Image.fromarray(img_colored))

        # Calcola il numero di righe e colonne per una matrice quadrata
        grid_size = math.ceil(math.sqrt(num_channels))  # Lato della matrice quadrata
        
        # Ottieni la dimensione di ciascuna immagine
        width, height = images[0].size
        combined_image = Image.new('RGB', (width * grid_size, height * grid_size))

        # Incolla le immagini nella posizione corretta della griglia
        for index, img in enumerate(images):
            row = index // grid_size
            col = index % grid_size
            combined_image.paste(img.convert('RGB'), (col * width, row * height))

        # Salva l'immagine combinata
        combined_image.save('/app/src/imgs/combined_image.png')
        
        # matplotlib.use('Agg')
        # # Assicurati di selezionare un campione
        # if sample.ndim == 3:  # Caso [C, H, W]
        #     num_channels = sample.shape[0]
        # elif sample.ndim == 4:  # Caso [N, C, H, W]
        #     num_channels = sample.shape[1]
        # else:
        #     raise ValueError("Dimensione del tensore non supportata.")

        # # Creare una figura per visualizzare i canali
        # fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))  # 1 riga e num_channels colonne

        # # Iterare sui canali e mostrarli
        # for i in range(num_channels):
        #     if sample.ndim == 4:
        #         axs[i].imshow(sample[0, i].numpy(), cmap='gray')  # Seleziona il primo campione
        #     else:
        #         axs[i].imshow(sample[i].numpy(), cmap='gray')  # Seleziona il canale

        #     axs[i].set_title(f'Canale {i + 1}')
        #     axs[i].axis('off')  # Nascondere gli assi

        # plt.tight_layout()
        # plt.show()


    
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