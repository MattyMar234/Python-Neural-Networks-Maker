
import sys
import torch
from torchvision import transforms

import psycopg2
import os

from DatasetComponents.DataModule.Munich480_DataModule import *

from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import *

from DatasetComponents.Datasets.munich480 import Munich480
from Networks.NetworkComponents.ImageSegmentationModels import UNET_2D
from Networks.NetworkFactory import *
from Networks.NetworkManager import *
from Networks.NetworkComponents.TrainingModel import *
from Networks.NetworkComponents.NeuralNetworkBase import *

import argparse 
from pathlib import Path


MODELS_OUTPUT_FOLDER = os.path.join(Path(os.getcwd()).parent.absolute(), 'Models')


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
    
    print(args)
    
    
    check_pytorch_cuda()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if device == "cuda" :
        deviceName = torch.cuda.get_device_name(device=None)
        
        if device.type == 'cuda' and deviceName == 'NVIDIA GeForce RTX 3060 Ti':
            print(f"set float32 matmul precision to \'medium\'")
            torch.set_float32_matmul_precision('medium')
            
            """You are using a CUDA device ('NVIDIA GeForce RTX 3060 Ti') that 
            has Tensor Cores. To properly utilize them, you should set 
            `torch.set_float32_matmul_precision('medium' | 'high')` which will 
            trade-off precision for performance. 
            For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"""
    
    
    
     
    datamodule: DataModuleBase = Munich480_DataModule(
        datasetFolder = "/dataset/munich480",
        batch_size=args.batch_size,
        num_workers=args.workers,
        year= Munich480.Year.Y2016,# | Munich480.Year.Y2017,
        distance= Munich480.Distance.m10 | Munich480.Distance.m20 | Munich480.Distance.m60,
    ) 
    
    
    
    # datamodule.setup()
    # train = datamodule.train_dataloader()
    # sample = train.dataset[0]
    
    # print(sample[0].shape)
    
    # train.dataset.show_sample(sample[0])
    
    # return 
    
    
    # UNET, trainer = NetworkFactory.makeNetwork(
    #     trainingModel = TrainModel_Type.ImageClassification,
    #     networkType=NetworkFactory.NetworkType.UNET_2D,
    #     in_channel=144,
    #     num_classes=27
    # )
    
    #print(datamodule.number_of_channels())
    
    UNET_2D_Model: ModelBase = UNET_2D(in_channel=datamodule.number_of_channels(), out_channel=datamodule.number_of_classes(), inputSize= datamodule.input_size())


    databaseParametre = DatabaseParametre(
        host="host.docker.internal",
        port="5432",
        database="IMAGES",
        user="postgres",
        password="admin",
        maxconn  = 10,
        timeout  = 10
    )
    
    
    
   

    
    

        
    

    
    
    
    
    # TRAINING_DATASET: ImageDataset_Postgres = ImageDataset_Postgres(
    #     imageSize=(32,32,3),
    #     classesCount=10,
    #     connectionParametre=databaseParametre,
    #     table=TrainingImages(),
    #     transform=training_trasforms1,
    #     oneHot=True,
    #     stackChannel=True
    # )
    
    # TEST_DATASET: ImageDataset_Postgres = ImageDataset_Postgres(
    #     imageSize=(32,32,3),
    #     classesCount=10,
    #     connectionParametre=databaseParametre,
    #     table=TestImages(),
    #     transform=test_trasforms1,
    #     oneHot=True,
    #     stackChannel=True
    # )
    
   
    
   
    
    networkManager = NetworkManager(
        device=device,
        model=UNET_2D_Model,
        args = args,
        workingFolder= MODELS_OUTPUT_FOLDER,
    )
    
    if args.train:
        networkManager.lightTrainNetwork(datamodule = datamodule)

    if args.test:
        #networkManager.lightTestNetwork(datamodule = datamodule)
    
        datamodule.setup()
        dataloader: DataLoader = datamodule.test_dataloader()#datamodule.test_dataloader()
        dataset = dataloader.dataset
        
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device(device))
        UNET_2D_Model.load_state_dict(checkpoint["state_dict"])
        UNET_2D_Model.to(device)
        UNET_2D_Model.eval()
        
        
        with torch.no_grad():
            idx: int = 0
            
            if args.idx:
                idx = args.idx % len(dataset)
            
            x, y = dataset[idx]
            x = x.to(device)
    
            y_hat = UNET_2D_Model(x.unsqueeze(0))
            
            datamodule.show_processed_sample(x, y_hat, y, idx)
    
    
    #networkManager.lightTestNetwork(testDataset=TEST_DATASET)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt_path',         type=Path,  default=None,           help='checkpoint or pretrained path')
    parser.add_argument('--ouputs',            type=Path,  default=MODELS_OUTPUT_FOLDER,  help='logs and data output path')
    parser.add_argument('--data_dir',          type=Path,  default=Path.cwd().parent)
    parser.add_argument('--dataset',           type=str,   default='?',            choices=['lombardia', 'munich'])
    parser.add_argument('--test_id',           type=str,   default='A',            choices=['A', 'Y'])
    parser.add_argument('--arch',              type=str,   default='?',            choices=['LaNet', 'AlexNet', 'VCC19', 'UNet'])
    parser.add_argument('-e', '--epochs',      type=int,   default=1)
    parser.add_argument('-b', '--batch_size',  type=int,   default=2)
    parser.add_argument('-w', '--workers',     type=int,   default=0)
    parser.add_argument('-d', '--gpu_or_cpu',  type=str,   default='gpu',          choices=['gpu', 'cpu'])
    parser.add_argument('--gpus',              type=int,   default=[0],            nargs='+')
    parser.add_argument('--idx',               type=int,  default=0)
    parser.add_argument('--test' ,      action='store_true')
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--compile',    action='store_true')

    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0 --ckpt_path /app/Models/UNET_2D/checkpoints/epoch=9-avg_val_loss=0.43453678.ckpt
     #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0
    
    # parser.add_argument("--devices", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=1)
    
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    
    if not args.test and not args.train:
        print("Please specify either --test or --train")
        sys.exit(1)
    
    if args.test and args.ckpt_path == None:
        print("Please specify checkpoint path with --ckpt_path")
        sys.exit(1)
    #print(type(args))
    
    main(args)
    
  