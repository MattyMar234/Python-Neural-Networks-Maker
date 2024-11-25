
import json
import logging.config
import sys
import torch
from torchvision import transforms

import psycopg2
import os

from DatasetComponents import DatamoduleFactory
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
from Networks.NetworkComponents.TrainingModel import *
from Networks.NetworkComponents.NeuralNetworkBase import *
from torch.utils.data import DataLoader

import argparse 
from pathlib import Path
import time
from Globals import *
from dotenv import load_dotenv

from Networks.NetworkManager import NetworkManager
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
    
    
    networkManager = NetworkManager(
        device=device,
        model=model,
        args = args,
        workingFolder= Globals.MODELS_TRAINING_FOLDER,
    )
    
    networkManager.lightTrainNetwork(datamodule = datamodule)


def testModel(args: argparse.Namespace | None, device: str, datamodule: DataModuleBase, model: ModelBase) -> None:
    pass



def exportModel(args: argparse.Namespace | None, model: ModelBase) -> None:
    pass
    



def main() -> None:
    
    setupLogging()
    load_dotenv()
    
    error: bool = False
    
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt_path',         type=Path,   default=None,            help='checkpoint or pretrained path')
    parser.add_argument('--ouputs',            type=Path,   default=Globals.MODELS_TRAINING_FOLDER,  help='logs and data output path')
    parser.add_argument('--data_dir',          type=Path,   default=Path.cwd().parent)
    parser.add_argument('--dataset',           type=str,    default='?',             choices=DatamoduleFactory.AvailableDatabodule.values())
    parser.add_argument('--test_id',           type=str,    default='A',             choices=['A', 'Y'])
    parser.add_argument('--arch',              type=str,    default='?',             choices= NetArchs.AvailableArchitetture.keys())
    parser.add_argument(f'--{Globals.EPOCHS}', type=int,    default=1)
    parser.add_argument('--batch_size',        type=int,    default=2)
    parser.add_argument('--workers',           type=int,    default=0)
    parser.add_argument('--gpu_or_cpu',        type=str,    default='gpu',           choices=['gpu', 'cpu'])
    parser.add_argument('--gpus',              type=int,    default=[0],             nargs='+')
    parser.add_argument('--idx',               type=int,    default=0)
    parser.add_argument(f'--{Globals.LOGGER_VERSION}', type=int,   default=Globals.AUTOMATIC_VERSIONANING_VALUE)
    
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
    
    parser.add_argument(f'--{Globals.DB_PORT}', type=str, default=str(os.environ['DB_PORT']))
    parser.add_argument(f'--{Globals.DB_HOST}', type=str, default=str(os.environ['DB_HOST']))
    parser.add_argument(f'--{Globals.DB_NAME}', type=str, default=str(os.environ['DB_NAME']))
    parser.add_argument(f'--{Globals.DB_USER}', type=str, default=str(os.environ['DB_USER']))
    parser.add_argument(f'--{Globals.DB_PASSWORD}', type=str, default=str(os.environ['DB_PASSWORD']))
    
    
    parser.add_argument(f'--{Globals.ENABLE_DATABASE}', action='store_true')
    parser.add_argument(f'--{Globals.EXPORT_MODEL}',    action='store_true')
    parser.add_argument('--summary',  action='store_true')
    parser.add_argument('--test' ,    action='store_true')
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--compile',  action='store_true')
    parser.add_argument('--work',     action='store_true')


    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0 --ckpt_path /app/Models/UNET_2D/checkpoints/epoch=9-avg_val_loss=0.43453678.ckpt
    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0
    #python main.py --train --worker 12 --batch_size 2 --epochs 40 --arch UNET_2D --ckpt_path /app/Models/UNET_2D/checkpoints/last.ckpt --lr=1e-3
    #python main.py --train --worker 12 --batch_size 2 --epochs 50 --arch UNET_2D --lr=1e-4 --sch=stepLR --gamma 0.99 --step_size=1 --step_type=epoch --ckpt_path=/app/Models/UNET_2D/checkpoints/last.ckpt --dataset=Munich_2D
    
    # parser.add_argument("--devices", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=1)
    
    # sudo -E python main.py --test --ckpt_path /app/Models/UNET_2D/checkpoints/last.ckpt --idx 2002
    
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    
    if not args.test and not args.train and not args.summary and not args.work:
        Globals.APP_LOGGER.error("Please specify at least one of the following operation options: --train, --test, --summary, --work")
        error = True
    
    if args.test and args.ckpt_path == None:
        Globals.APP_LOGGER.error("Please specify checkpoint path with --ckpt_path")
        error = True
    
    if args.arch == '?':
        Globals.APP_LOGGER.error("Please specify architecture with --arch")
        error = True 
        
    if args.dataset == '?':
        Globals.APP_LOGGER.error("Please specify dataset with --dataset")
        error = True 
         
    if error:
        sys.exit(1)
        
    argsAsDict = vars(args)
        
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
    
    
    # datamodule: Munich480_DataModule = Munich480_DataModule(
    #     datasetFolder = "/dataset/munich480",
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     useTemporalSize=False,
    #     year= Munich480.Year.Y2016
    # ) 
    
    
    
    
    
    datamodule: DataModuleBase = DatamoduleFactory.makeDatamodule(datasetName=args.dataset, args=args)
    NetworkModel: ModelBase = NetArchs.create_instance(args.arch, datamodule=datamodule, **vars(args))
    

    if args.summary:
        print(NetworkModel.makeSummary())
    
    if args.train:
        trainModel(args=args, device= device, datamodule= datamodule, model= NetworkModel)
    
    if args.test:
        testModel(args=args, device= device, datamodule = datamodule, model= NetworkModel)

    if args.work:
        datamodule.on_work(model= NetworkModel, device= device, **vars(args))

if __name__ == "__main__":
    main()
    
    
    
    
  