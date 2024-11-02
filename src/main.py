
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

    deviceName = torch.cuda.get_device_name(device=None)
    
    if device.type == 'cuda' and deviceName == 'NVIDIA GeForce RTX 3060 Ti':
        print(f"set float32 matmul precision to \'high\'")
        torch.set_float32_matmul_precision('high')
        
        """You are using a CUDA device ('NVIDIA GeForce RTX 3060 Ti') that 
        has Tensor Cores. To properly utilize them, you should set 
        `torch.set_float32_matmul_precision('medium' | 'high')` which will 
        trade-off precision for performance. 
        For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"""
     
    datamodule = Munich480_DataModule(
        datasetFolder = "/dataset/munich480",
        batch_size=args.batch_size,
        num_workers=args.workers,
        year= Munich480.Year.Y2016 | Munich480.Year.Y2017,
        distance= Munich480.Distance.m10 | Munich480.Distance.m20 | Munich480.Distance.m60,
    ) 
    
    datamodule.setup()
    train = datamodule.train_dataloader()
    
    
    # UNET, trainer = NetworkFactory.makeNetwork(
    #     trainingModel = TrainModel_Type.ImageClassification,
    #     networkType=NetworkFactory.NetworkType.UNET_2D,
    #     in_channel=144,
    #     num_classes=27
    # )
    
    print(datamodule.number_of_channels())
    
    UNET_2D_Model: ModelBase = UNET_2D(in_channel=datamodule.number_of_channels(), out_channel=datamodule.number_of_classes(), inputSize= datamodule.input_size())
    
    print(UNET_2D_Model.makeSummary())
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
    
    
    
    training_trasforms1 = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=15, fill=(255, 255, 255)),
        #transforms.Resize((572, 572)),
        transforms.ToTensor()
    ])
    
    validation_trasforms1 = transforms.Compose([
        #transforms.Resize((572, 572)),
        transforms.ToTensor()
    ])
    
    TRAINING_DATASET = Munich480(
        folderPath= os.path.join(Path(os.getcwd()).parent.absolute(), 'Data', 'Datasets','munich480'),
        mode= Munich480.DataType.TRAINING,
        year=Munich480.Year.Y2016,
        transforms=training_trasforms1,
    )
    
    VALIDATION_DATASET = Munich480(
        folderPath= os.path.join(Path(os.getcwd()).parent.absolute(), 'Data', 'Datasets','munich480'),
        mode= Munich480.DataType.VALIDATION,
        year=Munich480.Year.Y2016,
        transforms=validation_trasforms1
    )
    
    TEST_DATASET = Munich480(
        folderPath= os.path.join(Path(os.getcwd()).parent.absolute(), 'Data', 'Datasets', 'munich480'),
        mode= Munich480.DataType.VALIDATION,
        year=Munich480.Year.Y2016,
        transforms=validation_trasforms1
    )
    
    
    
    # checkpoint = torch.load(args.ckpt_path, weights_only=True)
    # UNET_model.load_state_dict(checkpoint["state_dict"])
    # UNET_model.eval()
    
    # with torch.no_grad():
        
    #     x, y = TEST_DATASET[0]
        
    #     print(x.shape)
        
    #     index = 4*20
        
        
    #     x_rgb = x[index:3 + index, :, :].permute(1, 2, 0) * 255  # Forma: [48, 48, 3]
    #     y_hat = UNET_model(x.unsqueeze(0))
        
    #     print(y_hat.shape, y.shape)
    #     y_hat_classes = torch.argmax(y_hat, dim=1).squeeze(0)  # Forma: [48, 48]
    #     mask = y_hat_classes.cpu().numpy()
    #     cmap = plt.get_cmap('tab20', 27)
    #     mask_rgb = cmap(mask)[:, :, :3]  # Ottieni i colori per ogni classe
    #     mask_rgb = (mask_rgb * 255).astype('uint8')  # Convertilo a uint8
    #     mask_alpha = (mask > 0).astype('float32')  # Imposta l'alpha in base alla maschera
    #     #print(y_hat.squeeze(0))
        
    #     overlay = (mask_rgb * mask_alpha[:, :, None]) / 255.0  # Sovrapposizione

    #     combined = (x_rgb / 255.0) * (1 - mask_alpha[:, :, None]) + overlay

    #     # Visualizza l'immagine combinata
    #     plt.imshow(combined)
    #     plt.title('Image with Segmentation Overlay')
    #     plt.axis('off')
    #     plt.show()


        # # Visualizza y e y_hat_classes
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # # Visualizza la classe reale
        # ax[0].imshow(y, cmap=cmap)  # o usa cmap se y ha più classi
        # ax[0].set_title('Ground Truth')
        # ax[0].axis('off')

        # # Visualizza la classe predetta
        # ax[1].imshow(y_hat_classes.cpu().numpy(), cmap=cmap)  # Usa la mappa di colori
        # ax[1].set_title('Predicted')
        # ax[1].axis('off')

        
    

    
    
    
    
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
        model=UNET_model,
        args = args,
        workingFolder= MODELS_OUTPUT_FOLDER,
        # ModelWeights_Input_File  = None,
        # ModelWeights_Output_File = f'{UNET_model.__class__.__name__}_v1.pth',
        # ckpt_file                = f'{UNET_model.__class__.__name__}_v1.ckpt'
    )
    
    
    
    
    networkManager.lightTrainNetwork(
        trainingDataset=TRAINING_DATASET,#TRAINING_DATASET,
        testDataset=VALIDATION_DATASET,
    )
    
    #networkManager.lightTestNetwork(testDataset=TEST_DATASET)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt_path',          type=Path,  default=None,           help='checkpoint or pretrained path')
    parser.add_argument('--ouputs',             type=Path,  default=MODELS_OUTPUT_FOLDER,  help='logs and data output path')
    parser.add_argument('--data_dir',           type=Path,  default=Path.cwd().parent)
    parser.add_argument('--dataset',            type=str,   default='?',            choices=['lombardia', 'munich'])
    parser.add_argument('--test_id',            type=str,   default='A',            choices=['A', 'Y'])
    parser.add_argument('--arch',               type=str,   default='swin_unetr',   choices=['LaNet', 'AlexNet', 'VCC19', 'UNet'])
    parser.add_argument('--compile',            type=int,   default= 0, choices=[0, 1])
    parser.add_argument('-e' , '--epochs',      type=int,   default=1)
    parser.add_argument('-bs','--batch_size',   type=int,   default=2)
    parser.add_argument('-w' ,'--workers',      type=int,   default=0)
    parser.add_argument('--gpu_or_cpu',         type=str,   default='gpu',          choices=['gpu', 'cpu'])
    parser.add_argument('--gpus',               type=int,   default=[0],            nargs='+')

    #python main.py --workers 7 --batch_size 2 --epochs 12 --compile 0 --ckpt_path /app/Models/UNET_2D/checkpoints/epoch=9-avg_val_loss=0.43453678.ckpt
    
    
    # parser.add_argument("--devices", type=int, default=0)
    # parser.add_argument("--epochs", type=int, default=1)
    
    if '--help' in sys.argv or '-help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    #print(type(args))
    
    main(args)
    
  