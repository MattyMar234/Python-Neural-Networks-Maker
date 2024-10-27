import torch
from torchvision import transforms

import psycopg2
import os

from Dataset.ImageDataset import *
from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import *

from Networks.NetworkFactory import *
from Networks.NetworkManager import *
from Networks.TrainingModel import *


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

def main():
    
    global MODELS_OUTPUT_FOLDER
    
    check_pytorch_cuda()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        trainingDataset=TRAINING_DATASET,
        testDataset=TEST_DATASET,
        epochs=2,
        batchSize=4,
        workers=5
    )
    

if __name__ == "__main__":
    main()
    
  