
import json
import logging.config
import sys
import torch
from torchvision import transforms

import psycopg2
import os

from DatasetComponents.DataModule.Munich480_DataModule import *

from Database.DatabaseConnection import PostgresDB
from Database.DatabaseConnectionParametre import DatabaseParametre
from Database.Tables import *

from DatasetComponents.Datasets.DatasetBase import DatasetBase
from DatasetComponents.Datasets.munich480 import Munich480
import Globals
from LoggerSetup import setupLogging
from Networks.Architettures.SemanticSegmentation.UNet import UNET_2D
import Networks.Architettures as NetArchs
from Networks.Metrics.ConfusionMatrix import *
from Networks.NetworkManager import *
from Networks.NetworkComponents.TrainingModel import *
from Networks.NetworkComponents.NeuralNetworkBase import *

import argparse 
from pathlib import Path
import time
from Globals import *
from dotenv import load_dotenv

from Utility.TIF_creator import TIF_Creator
import logging.config
import logging







def check_pytorch_cuda() -> bool:
    Globals.APP_LOGGER.info(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        Globals.APP_LOGGER.info("CUDA is available.")
        Globals.APP_LOGGER.info(f"Device found: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            Globals.APP_LOGGER.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        return True
    
    else:
        Globals.APP_LOGGER.info("CUDA is not available.")
        return False

def trainModel(args: argparse.Namespace | None, device: str, datamodule: DataModuleBase, model: ModelBase) -> None:
    global MODELS_OUTPUT_FOLDER
    
    networkManager = NetworkManager(
        device=device,
        model=model,
        args = args,
        workingFolder= MODELS_OUTPUT_FOLDER,
    )
    
    networkManager.lightTrainNetwork(datamodule = datamodule)

def testModel(args: argparse.Namespace | None, device: str, datamodule: DataModuleBase, model: ModelBase) -> None:
    


    # databaseParametre = DatabaseParametre(
    #     host="host.docker.internal",
    #     port="5432",
    #     database="IMAGES",
    #     user="postgres",
    #     password="admin",
    #     maxconn  = 10,
    #     timeout  = 10
    # )
    
    
    
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
    
   
    
   
    
    
    
    
        

    
        #networkManager.lightTestNetwork(datamodule = datamodule)
    
    datamodule.setup()
    dataloader: DataLoader = datamodule.test_dataloader()#datamodule.test_dataloader()
    dataset:DatasetBase = dataloader.dataset
    
    
    confMatrix: ConfusionMatrix  = ConfusionMatrix(classes_number = 27, ignore_class=datamodule.classesToIgnore(), mapFuntion=datamodule.map_classes)
    
    
    checkpoint = torch.load(args.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    creator:TIF_Creator = TIF_Creator('/app/geoData')
  
  
        
    if args.idx >= 0:
        idx = args.idx % len(dataset)

        with torch.no_grad():  
            data: Dict[str, any] = dataset.getItems(idx)
            
            x = data['x']
            y = data['y']
            y = y.unsqueeze(0)
            x = x.to(device)
            
            y_hat = model(x.unsqueeze(0))
            
            y_hat_ = torch.argmax(y_hat, dim=1)
            y_ = torch.argmax(y, dim=0)
            
            print(y_hat_.shape, y_.shape)
            
            y_ = y_.cpu().detach()
            y_hat_ = y_hat_.cpu().detach()
            
            
            
            confMatrix.update(y_pr=y_hat_, y_tr=y_)
            _, graphData = confMatrix.compute(showGraph=False)
            
            confMatrix.reset()
        
            datamodule.show_processed_sample(x, y_hat, y, idx, graphData)
            
            #creator.makeTIF(f'{idx}.tif', data['profile'], data =y_hat_, classColorMap = Munich480_DataModule.MAP_COLORS_AS_RGB_LIST)

        return
    
    if args.idx == -1:
        
        creator.mergeTIFs('/app/merged.tif')
        return
        
        with torch.no_grad(): 
            for idx in range(len(dataset)):
                data: Dict[str, any] = dataset.getItems(idx)

                print(f"Processing {idx}/{len(dataset)}")

                x = data['x']
                y = data['y']
                x = x.to(device)

                y_hat = model(x.unsqueeze(0))

                y_hat_ = torch.argmax(y_hat, dim=1)
                y_ = torch.argmax(y, dim=0)


                y_hat_RGB = np.zeros((3, 48, 48), dtype=np.uint8)
                
                for i in range(48):
                    for j in range(48):
                        class_id = int(y_hat_[0, i, j])
                        y_hat_RGB[:, i, j] = Munich480_DataModule.MAP_COLORS_AS_RGB_LIST[class_id]

                creator.makeTIF(f'{idx}.tif', data['profile'], data =y_hat_RGB, channels = 3, width=48, height=48) 
            
            
            
        
        
        
        
        
        
        # confMatrix.reset()
        
        # datamodule.show_processed_sample(x, y_hat, y, idx, graphData)
    
    
    #networkManager.lightTestNetwork(testDataset=TEST_DATASET)

def main() -> None:
    
    setupLogging()
    load_dotenv()
    
    error: bool = False
    
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt_path',         type=Path,   default=None,            help='checkpoint or pretrained path')
    parser.add_argument('--ouputs',            type=Path,   default=MODELS_OUTPUT_FOLDER,  help='logs and data output path')
    parser.add_argument('--data_dir',          type=Path,   default=Path.cwd().parent)
    parser.add_argument('--dataset',           type=str,    default='?',             choices=['lombardia', 'munich'])
    parser.add_argument('--test_id',           type=str,    default='A',             choices=['A', 'Y'])
    parser.add_argument('--arch',              type=str,    default='?',             choices= NetArchs.AvailableArchitetture.keys())
    parser.add_argument('-e', f'--{Globals.EPOCHS}',   type=int,    default=1)
    parser.add_argument('-b', '--batch_size',  type=int,    default=2)
    parser.add_argument('-w', '--workers',     type=int,    default=0)
    parser.add_argument('-d', '--gpu_or_cpu',  type=str,    default='gpu',           choices=['gpu', 'cpu'])
    parser.add_argument('--gpus',              type=int,    default=[0],             nargs='+')
    parser.add_argument('--idx',               type=int,    default=0)
    
    parser.add_argument(f'--{Globals.LEARNING_RATE}',       type=float, default=1e-4,            help='learning rate')
    parser.add_argument(f'--{Globals.SCHEDULER_TYPE}',      type=str,   default=ShedulerType.NONE, choices=ShedulerType.values())
    parser.add_argument(f'--{Globals.SCHEDULER_GAMMA}',     type=float, default=1.0)
    parser.add_argument(f'--{Globals.SCHEDULER_STEP_SIZE}', type=int,   default=1)
    parser.add_argument(f'--{Globals.SCHEDULER_STEP_TYPE}', type=str,   default='epoch', choices=SCHEDULER_STEP_TYPE_AVAILABLE)
    parser.add_argument(f'--{Globals.T_MAX}',               type=int,   default=10)
    parser.add_argument(f'--{Globals.T_MULT}',              type=float, default=1.0)
    parser.add_argument(f'--{Globals.ETA_MIN}',             type=float, default=0.0)
    parser.add_argument(f'--{Globals.FACTOR}',              type=float, default=1.0)
    parser.add_argument(f'--{Globals.START_FACTOR}',        type=float, default=1)
    parser.add_argument(f'--{Globals.END_FACTOR}',          type=float, default=1)
    parser.add_argument(f'--{str(Globals.MILESTONES)}',     type=str,   default=[10], nargs='+')
    parser.add_argument(f'--{Globals.POWER}',               type=float, default=1)
    
    parser.add_argument('--summary',    action='store_true')
    parser.add_argument('--test' ,      action='store_true')
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--compile',    action='store_true')

    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0 --ckpt_path /app/Models/UNET_2D/checkpoints/epoch=9-avg_val_loss=0.43453678.ckpt
    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0
    #python main.py --train --worker 12 --batch_size 2 --epochs 40 --arch UNET_2D --ckpt_path /app/Models/UNET_2D/checkpoints/last.ckpt --lr=1e-3
    
    # parser.add_argument("--devices", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=1)
    
    # sudo -E python main.py --test --ckpt_path /app/Models/UNET_2D/checkpoints/last.ckpt --idx 2002
    
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    
    if not args.test and not args.train and not args.summary:
        Globals.APP_LOGGER.error("Please specify either --test, --train or --summary")
        error = True
    
    if args.test and args.ckpt_path == None:
        Globals.APP_LOGGER.error("Please specify checkpoint path with --ckpt_path")
        error = True
    
    if not args.arch:
        Globals.APP_LOGGER.error("Please specify architecture with --arch")
        error = True  
         
    if error:
        sys.exit(1)
        
    Globals.APP_LOGGER.info(args)
    
    device: torch.device = torch.device("cuda" if args.gpu_or_cpu == 'gpu' and check_pytorch_cuda() else "cpu")
    Globals.APP_LOGGER.info(f"Device selected: {device}") 
    
    if device.type == 'cuda' :
        deviceName = torch.cuda.get_device_name(device=None)
        
        if deviceName == 'NVIDIA GeForce RTX 3060 Ti':
            Globals.APP_LOGGER.info(f"set float32 matmul precision to \'medium\'")
            torch.set_float32_matmul_precision('medium')
            
            """You are using a CUDA device ('NVIDIA GeForce RTX 3060 Ti') that 
            has Tensor Cores. To properly utilize them, you should set 
            `torch.set_float32_matmul_precision('medium' | 'high')` which will 
            trade-off precision for performance. 
            For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"""
    
    
    datamodule: Munich480_DataModule = Munich480_DataModule(
        datasetFolder = "/dataset/munich480",
        batch_size=args.batch_size,
        num_workers=args.workers,
        useTemporalSize=False,
        year= Munich480.Year.Y2016
    ) 
    
    
    
    
    
    
    NetworkModel: ModelBase = NetArchs.create_instance(args.arch, datamodule=datamodule, **vars(args))
    
    #UNET_2D_Model: ModelBase = UNET_2D(in_channel=datamodule.number_of_channels(), out_channel=datamodule.number_of_classes(), inputSize= datamodule.input_size())
    
    if args.summary:
        print(NetworkModel.makeSummary())
    
    if args.train:
        trainModel(args=args, device= device, datamodule= datamodule, model= NetworkModel)
    
    if args.test:
        testModel(args=args, device= device, datamodule = datamodule, model= NetworkModel)

if __name__ == "__main__":
    main()
    
    
    
    
  