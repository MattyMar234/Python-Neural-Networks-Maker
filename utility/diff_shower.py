import math
import re
import sys

import matplotlib.pyplot as plt
import rasterio
import argparse 
import numpy as np
import os
from PIL import Image


OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")



def normalizza_banda(banda_img):
    """
    Normalizza una banda dividendo i suoi valori per il massimo valore, portando i valori nel range [0, 1].
    
    Parametri:
    - banda_img: l'immagine della banda come array numpy.
    
    Ritorna:
    - L'immagine normalizzata.
    """
    return (banda_img - banda_img.min()) / (banda_img.max() - banda_img.min())



    
    
def load_data(file_path:str) -> np.array:
    
    with rasterio.open(file_path) as src:
        print(f"Bande: {src.count}")
        data = src.read()
        dType = src.dtypes
    
        data_normalized = [((band - np.min(band)) / (np.max(band) - np.min(band)) * 255).astype(np.uint8) for band in data]
    #data = np.loadtxt(file_path)
    #data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
    
        for band in data:
            data_normalized.append(band)
    
    return data_normalized

def combine_bands_as_rgb(bands: list[np.array]) -> np.array:
    if len(bands) < 3:
        raise ValueError("Non ci sono abbastanza bande per formare un'immagine RGB.")
    
    # Assumiamo che le bande 1, 2, e 3 siano da usare come R, G, B rispettivamente
    rgb_image = np.zeros((bands[0].shape[0], bands[0].shape[1], 3), dtype=np.uint8)
    rgb_image[..., 0] = bands[0]  # Banda 1 come R
    rgb_image[..., 1] = bands[1]  # Banda 2 come G
    rgb_image[..., 2] = bands[2]  # Banda 3 come B
    
    return rgb_image


def display_with_matplotlib(file_list: list[str], num_bands: int = 8) -> None:
    
    colorList = ["Reds", "Greens", "Blues" , "inferno"]*2
    
    total_plots = len(file_list) * (num_bands + 1)
    cols = math.ceil(math.sqrt(total_plots ))  # Numero di colonne per la griglia
    rows = math.ceil(total_plots  / cols)      # Numero di righe per la griglia
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Appiattisce gli assi in una lista per un accesso facile
    
    plot_index = 0
    for i, file in enumerate(file_list):
        bands: list[np.array] = load_data(file)
        
        for band_index in range(min(num_bands, len(bands))):
            axes[plot_index].imshow(bands[band_index], cmap= colorList[band_index])  # Visualizza la banda corrente
            axes[plot_index].set_title(f"{os.path.basename(file)} - Banda {band_index + 1}")
            axes[plot_index].axis("off")  # Rimuove gli assi per una visualizzazione più pulita
            plot_index += 1
            
        # Visualizza l'immagine combinata RGB
        if len(bands) >= 3:  # Solo se ci sono almeno 3 bande
            rgb_image = combine_bands_as_rgb(bands)
            axes[plot_index].imshow(rgb_image)
            axes[plot_index].set_title(f"{os.path.basename(file)} - RGB")
            axes[plot_index].axis("off")
            plot_index += 1

      
       
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show() 
    
    

def dumpData(file_list: list[str]) :
    
    
    for f in file_list:
        data = load_data(f)
        
        for i, c in enumerate(data):
            print(f"{os.path.basename(f)}-{i}:")
            print(c)
            print(c.shape)

def fileInfo(file_list: list[str]):
    
    for f in file_list:
        print(f"{os.path.basename(f)}:")
        with rasterio.open(f) as src:
            print("Nome del file:", src.name)
            print("Driver:", src.driver)
            print("Altezza:", src.height)
            print("Larghezza:", src.width)
            print("Numero di bande:", src.count)
            print("Tipo di dati:", src.dtypes)

def main(args: argparse.Namespace | None) -> None:
    
    data_list = [file_path for file_path in args.paths]
    
    match(args.mode):
        case "graph" :
            display_with_matplotlib(data_list)
        case "dump":
            dumpData(data_list)
            
        case "info":
            fileInfo(data_list)
                
        case _ :
            raise 
    
   

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Leggi e visualizza dati da un file.")
    parser.add_argument("--paths", "-p", nargs='+',type=str, help="Percorso del file.")
    parser.add_argument("--mode", "-m", choices=["dump","graph", "file", "info"], default="graph", help= "modalità di visualizzazione.")
         
         
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
      
    args = parser.parse_args()
    main(args)               


#python diff_shower.py --paths /app/Data/Datasets/munich480/data16/1/20160103_10m.tif /app/Data/Datasets/munich480/data16/1/20160113_10m.tif /app/Data/Datasets/munich480/data16/1/20160121_10m.tif /app/Data/Datasets/munich480/data16/1/20160128_10m.tif /app/Data/Datasets/munich480/data16/1/20160212_10m.tif /app/Data/Datasets/munich480/data16/1/20160311_10m.tif /app/Data/Datasets/munich480/data16/1/20160128_10m.tif /app/Data/Datasets/munich480/data16/1/20160320_10m.tif /app/Data/Datasets/munich480/data16/1/20160323_10m.tif