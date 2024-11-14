import os
import numpy as np
import rasterio
from rasterio.merge import merge
import torch
import glob

#def makeTIF(self, baseProfile, dtype: str = 'uint8', channels: int = 1, width: int = 48, height: int = 48) -> None:


class TIF_Creator:
    def __init__(self, workingDir: str):
        self.output_folder = workingDir
        self.profile = None  # Profilo iniziale del mosaico
        self.is_initialized = False  # Controllo per inizializzare il file solo la prima volta

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def makeTIF(self, fileName: str, baseProfile: dict, data: torch.Tensor | np.ndarray | None = None, dtype: str = 'uint8', channels: int = 1, width: int = 48, height: int = 48) -> None:
        assert data is not None, "Data is None"
        
        if type(data) == torch.Tensor:
            data = data.cpu().numpy()
        
        shape = data.shape
        
        assert len(shape) == 3, f"Data must be 3D. Got {shape}"
        assert shape[0] == channels and shape[1] == width and shape[2] == height, f"Data must have this format: [{channels}, {width}, {height}]. Got {shape}"
        

        # Imposta il profilo e crea il file di output
        self.profile = baseProfile.copy()
        self.profile.update({
            'dtype': dtype,
            'count': channels,
            'width': width,
            'height': height,
        })

        with rasterio.open(os.path.join(self.output_folder, fileName), 'w', **self.profile) as dst:
            for i in range(1, channels + 1):
                dst.write(data[i - 1], i)
        
    def mergeTIFs(self, output_file: str = 'mosaico_finale.tif') -> None:
        # Trova tutti i file TIF creati per le tessere
        tile_files = glob.glob(os.path.join(self.output_folder, '*.tif'))
        
        
        # Apri tutti i file TIF
        datasets = [rasterio.open(tile) for tile in tile_files]
        
        # Unisci i dataset per creare il mosaico
        mosaic, out_transform = merge(datasets)
        
        # Usa il profilo della prima tessera come base per il mosaico
        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })
        
        # Scrivi il mosaico finale in un nuovo file
        with rasterio.open(output_file, 'w', **out_meta) as dest:
            dest.write(mosaic)
        
        print(f"Mosaico finale creato: {output_file}")