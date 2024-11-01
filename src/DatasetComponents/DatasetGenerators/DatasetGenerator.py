from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
import csv
import enum
import enum
from Database.Tables import *
from Database.DatabaseConnection import PostgresDB

exclude_fonts = [
    'Noto', 
    'Amiri', 
    'Segoe',
    'Kacst', 
    'KacstBook', 
    'Marlett', 
    'HoloLens MDL2 Assets', 
    'HoloLens', 
    'Segoe MDL2 Assets', 
    'EmojiOneColor-SVGinOT',
    'Symbol',
    'Webdings',
    'Wingding',
    'Himalaya',
    'DavidCLM-Bold',
    'DavidCLM-BoldItalic',
    'DavidCLM-Medium',
    'DavidCLM-MediumItalic',
    'NachlieliCLM-Bold',
    'NachlieliCLM-BoldOblique',
    'NachlieliCLM-Light',
    'NachlieliCLM-LightOblique',
    'FrankRuehlCLM-Bold',
    'FrankRuehlCLM-BoldOblique',
    'FrankRuehlCLM-Medium',
    'FrankRuehlCLM-MediumOblique'
    ]



def get_system_fonts():
    fonts = []
    
    # Percorsi di font per diversi sistemi operativi
    if os.name == 'nt':  # Windows
        fonts_dir = 'C:\\Windows\\Fonts'
    elif os.name == 'posix':
        # Controlla se è macOS o Linux
        if os.uname().sysname == 'Darwin':
            fonts_dir = '/Library/Fonts'
        else:
            fonts_dir = '/usr/share/fonts'
    else:
        raise Exception("Unsupported OS")

    # Aggiungere tutti i file .ttf e .otf dalla directory dei font
    for root, dirs, files in os.walk(fonts_dir):
        for file in files:
            if file.lower().endswith(('.ttf', '.otf')):
                font_name = os.path.splitext(file)[0]
                # Aggiungere il font solo se non contiene parole da escludere
                if not any(keyword.lower() in font_name.lower() for keyword in exclude_fonts):
                    fonts.append(os.path.join(root, file))

    return fonts

class DatabaseFormat(enum.Enum):
    CSV = 1
    PNG = 2
    JPG = 3
    BMP = 4
    TIFF = 5
    GIF = 6
    TAR = 7
    ZIP = 8
    

class DataGenerator:

    SIZE = (28, 28)
    FONT_SIZE = (5, 5)
    FONT_LIST = get_system_fonts()

    def __init__(self, scale: int, categoryImageCount: int = 1, maxRotation: float = 0.0, noiseLevel: tuple[int, int] = (0, 0)):
        self.__scale: int = scale
        self.__categoryImageCount: int = categoryImageCount
        self.__maxRotation: float = maxRotation
        self.__noiseLevel: tuple[int, int] = noiseLevel


    def generateOnPostgres(self, database: PostgresDB, table: _ImageTable) -> None:
        assert type(database) == PostgresDB, "Database must be a PostgresDB instance"

        database.execute_query(table.dropTableQuery())
        database.execute_query(table.createTableQuery())

        if not database.is_connected():
            database.connect()

        for category in range(10):
            for _ in range(self.__categoryImageCount):
                r_ch, g_ch, b_ch, _ = self._generateImage(
                    scale = self.__scale, 
                    number= category, 
                    rotation = random.random() * self.__maxRotation, 
                    noiseLevel=random.randint(self.__noiseLevel[0], self.__noiseLevel[1])
                )
                
                database.execute_query(table.generateInsertQuery(label=str(category), rCh=r_ch, gCh=g_ch, bCh=b_ch))

            


    def generate(self, outputFolder: str, format: DatabaseFormat):
        assert type(format) == DatabaseFormat, "Format must be a DatabaseFormat enum"
        
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        match format:
            case DatabaseFormat.CSV: 
        
                # File CSV per ogni canale
                r_csv = open(os.path.join(outputFolder, 'R_channel.csv'), mode='w', newline='')
                g_csv = open(os.path.join(outputFolder, 'G_channel.csv'), mode='w', newline='')
                b_csv = open(os.path.join(outputFolder, 'B_channel.csv'), mode='w', newline='')

                r_writer = csv.writer(r_csv)
                g_writer = csv.writer(g_csv)
                b_writer = csv.writer(b_csv)

                header = ['label'] + [f'pixel_{i}' for i in range(int((DataGenerator.SIZE[0]*self.__scale) * (DataGenerator.SIZE[1]*self.__scale)))]
                r_writer.writerow(header)
                g_writer.writerow(header)
                b_writer.writerow(header)
        
                for category in range(10):
                    for _ in range(self.__categoryImageCount):
                        r_ch, g_ch, b_ch, _ = self._generateImage(
                            scale = self.__scale, 
                            number= category, 
                            rotation = random.random() * self.__maxRotation, 
                            noiseLevel=random.randint(self.__noiseLevel[0], self.__noiseLevel[1])
                        )
                        
                        r_writer.writerow([category] + r_ch.tolist())
                        g_writer.writerow([category] + g_ch.tolist())
                        b_writer.writerow([category] + b_ch.tolist())
                        
                        #self._save_image_to_csv(npImage, category, r_writer, g_writer, b_writer)

                r_csv.close()
                g_csv.close()
                b_csv.close()
        
        
            
    def _generateImage(self, scale: int, number, rotation: float = 0.0, noiseLevel: int = 0):
        img = Image.new('RGB', (DataGenerator.SIZE[0] * scale, DataGenerator.SIZE[1] * scale), 
                        (int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)))

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(random.choice(DataGenerator.FONT_LIST), 24 * scale)
        
        text = str(number)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (img.size[0] - text_width) // 2
        text_y = (img.size[1] - text_height) // 2

        correction_x = (img.size[0] - text_width) * -0.05
        correction_y = (img.size[1] - text_height) * 0.375
        text_x -= int(correction_x)
        text_y -= int(correction_y)

        
        temp_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((text_x, text_y), text, fill=(int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)), font=font)
        
        # Applicare la rotazione
        temp_img = temp_img.rotate(rotation, expand=1)

        temp_x = (img.size[0] - temp_img.size[0]) // 2
        temp_y = (img.size[1] - temp_img.size[1]) // 2

        img.paste(temp_img, (temp_x, temp_y), temp_img)

        npImg = np.array(img)

      
        if noiseLevel > 0:
            npImg = self.__add_noise(npImg, noiseLevel)


        # separo i canali 
        r_channel = npImg[:, :, 0].flatten()
        g_channel = npImg[:, :, 1].flatten()
        b_channel = npImg[:, :, 2].flatten()

        return r_channel, g_channel, b_channel, font.getname()

    def __add_noise(self, img: np.array, noiseLevel: int) -> np.array:
        noise = np.random.normal(0, noiseLevel, img.shape)
        np_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return np_img
    
    
        
        

       
        
        

if __name__ == "__main__":
    dg = DataGenerator()
    dg.generate(scale = 1, outputFolder = "TEMP_CSV", categoryImageCount = 200, maxRotation=25, noiseLevel=(0, 10))