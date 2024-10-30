
import sys
import torch
from torchvision import transforms

import psycopg2
import os

from Dataset.ImageDataset import *
from Dataset.munich480 import *

from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import *

from Networks.NetworkFactory import *
from Networks.NetworkManager import *
from Networks.TrainingModel import *

import argparse 
from pathlib import Path


MODELS_OUTPUT_FOLDER = os.path.join(os.getcwd(), 'Models')


def check_pytorch_cuda() -> bool:
    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available. Device count: ", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("CUDA is not available.")
        return False


def main(args: argparse.Namespace | None) -> None:
    
    global MODELS_OUTPUT_FOLDER
    
    check_pytorch_cuda()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET = Munich480(
        folderPath= os.path.join(Path(os.getcwd()).parent.absolute(), 'Data', 'Datasets','munich480'),
        mode= Munich480.DataType.TRAINING,
        year=Munich480.Year.Y2016,
        transforms=None
    )
    
    print(len(DATASET))
    DATASET.visualize_sample(1000)
    
    for i in range(len(DATASET)):
        print(DATASET[i][0].shape)
        #DATASET[i]
        break
        pass
    
    # print(len(DATASET))
    # print(DATASET[1].shape)
    
    return


    databaseParametre = DatabaseParametre(
        host="host.docker.internal",
        port="5432",
        database="IMAGES",
        user="postgres",
        password="admin",
        maxconn  = 10,
        timeout  = 10
    )
    
    
    # training_trasforms1 = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     #transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    
    # test_trasforms1 = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    
    
    training_trasforms1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
        #transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_trasforms1 = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    TRAINING_DATASET: ImageDataset_Postgres = ImageDataset_Postgres(
        imageSize=(32,32,3),
        classesCount=10,
        connectionParametre=databaseParametre,
        table=TrainingImages(),
        transform=training_trasforms1,
        oneHot=True,
        stackChannel=True
    )
    
    TEST_DATASET: ImageDataset_Postgres = ImageDataset_Postgres(
        imageSize=(32,32,3),
        classesCount=10,
        connectionParametre=databaseParametre,
        table=TestImages(),
        transform=test_trasforms1,
        oneHot=True,
        stackChannel=True
    )
    
   
    
   
    
    AlexNet, trainer = NetworkFactory.makeNetwork(
        trainingModel = TrainModel_Type.ImageClassification,
        networkType=NetworkFactory.NetworkType.LaNet5_ReLU,
        in_channel=3,
        num_classes=10
        
    )
    
    networkManager = NetworkManager(
        device=device,
        model=AlexNet,
        workingFolder=os.path.join(MODELS_OUTPUT_FOLDER, f'{AlexNet.__class__.__name__}'),
        ModelWeights_Input_File  = None,
        ModelWeights_Output_File = f'{AlexNet.__class__.__name__}_v1.pth',
        ckpt_file                = f'{AlexNet.__class__.__name__}_v1.ckpt'
    )
    
    
    
    
    networkManager.lightTrainNetwork(
        traiableModel=trainer,
        trainingDataset=TEST_DATASET,#TRAINING_DATASET,
        testDataset=TEST_DATASET,
        epochs=2,
        batchSize=4,
        workers=5
    )



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt_path',          type=Path,  default=None,           help='checkpoint or pretrained path')
    parser.add_argument('--data_dir',           type=Path,  default=Path.cwd().parent)
    parser.add_argument('--dataset',            type=str,   default='?',            choices=['lombardia', 'munich'])
    parser.add_argument('--test_id',            type=str,   default='A',            choices=['A', 'Y'])
    parser.add_argument('--arch',               type=str,   default='swin_unetr',   choices=['LaNet', 'AlexNet', 'VCC19', 'UNet'])
    parser.add_argument('-e' , '--epochs',      type=int,   default=1)
    parser.add_argument('-bs','--batch_size',   type=int,   default=2)
    parser.add_argument('-w' ,'--workers',      type=int,   default=0)
    parser.add_argument('--gpu_or_cpu',         type=str,   default='gpu',          choices=['gpu', 'cpu'])
    parser.add_argument('--gpus',               type=int,   default=[0],            nargs='+')

    
    
    # parser.add_argument("--devices", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=1)
    
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    #print(type(args))
    
    main(args)
    
  