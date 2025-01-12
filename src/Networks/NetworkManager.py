import argparse
from dataclasses import dataclass
from torchvision import transforms
from torch import nn
import torch
from torchmetrics import Metric
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor

from DatasetComponents.DataModule import DataModuleBase

from .NetworkComponents.TrainingModel import *

from PIL import Image
import logging
from tqdm import tqdm
import os
import tqdm
import numpy as np
import shutil





class NetworkManager:

    __trainingRandomSeed: int = 42

    def __init__(self, device: torch.device, model: L.LightningModule, workingFolder: str | None = None, args: argparse.Namespace | None = None):
        
        assert model != None, "Model cannot be None"
        
        self._device: torch.device = device
        self._model:L.LightningModule = model
        self._args: argparse.Namespace | None = args
        
        trainFolderName = f"{model.__class__.__name__}-{args.dataset}"
        
        if workingFolder == None or workingFolder == "":
            self._workingFolder = os.path.join(os.getcwd(), trainFolderName)
        else:
            self._workingFolder = os.path.join(workingFolder, trainFolderName)
        
        if not os.path.exists(self._workingFolder):
            os.makedirs(self._workingFolder)
            
        
        self._model.to(self._device)
        
        
    
    @property
    def parameters(self):
        return self._model.parameters()
    
    @property
    def modelInfo(self):
        return self._model.__str__()
    
    
    def load_checkpoint(self, checkpoint_Path:str) -> None:
        self._model.load_from_checkpoint(checkpoint_Path)
    
   
    
    

    def _exportWeights(self, path:str):
        torch.save(self._model.state_dict(), path)
        logging.info(f'Modello salvato in {path}')
    

    def _importWeights(self, path:str):
        if os.path.isfile(path):
            self._model.load_state_dict(torch.load(path))
            logging.info(f'Modello {path} importato con successo')
        else:
            logging.error(f'File {path} non trovato')
            raise FileNotFoundError(f'File {path} non trovato')


    def lightTrainNetwork(self, datamodule: DataModuleBase, **kwargs):
        pl.seed_everything(NetworkManager.__trainingRandomSeed, workers= True)#self._args.workers > 0)
        
        
        
        tensorBoard_logger = TensorBoardLogger(
            save_dir= self._workingFolder, 
            name="TensorBoard_logs",
            version= vars(self._args).get(Globals.LOGGER_VERSION, None)
        )
        
        CSV_logger = CSVLogger(
            save_dir= self._workingFolder, 
            name="CSV_logs",
            version= vars(self._args).get(Globals.LOGGER_VERSION, None)    
        )
        
        tensorBoard_logger.log_hyperparams = lambda *args, **kwargs: None
        CSV_logger.log_hyperparams = lambda *args, **kwargs: None
        
    
        
        checkpoint_callback = ModelCheckpoint(
            monitor=TraingBase.AVG_VALIDATION_LOSS_LABEL_NAME,  # La metrica da monitorare
            dirpath=os.path.join(self._workingFolder, 'checkpoints'),  # Sottocartella per i checkpoint
            filename='{epoch}-{' + f'{TraingBase.AVG_VALIDATION_LOSS_LABEL_NAME}'+':.8f}',  # Nome del file
            save_top_k=2,  # Salva i migliori 3 checkpoint
            mode='min',  # Se la metrica deve essere minimizzata
            every_n_epochs=1,
            save_last=True,
        )
        
        trainer: pl.Trainer | None = None
        
        if self._device == torch.device("cuda"):
            if len(self._args.gpus) > 1:
                #raise NotImplementedError("Multi-GPU training is not implemented yet")
                strategy = "ddp"
            
                Globals.APP_LOGGER.info(f"Training Using Multi-GPU: {self._args.gpus} with strategy: {strategy}")
            
                trainer = pl.Trainer(
                    accelerator="gpu",
                    strategy="ddp", 
                    devices=self._args.gpus,
                    max_epochs=self._args.epochs,
                    min_epochs=1, 
                    #profiler="simple", #profiler="advanced"
                    #default_root_dir= self._workingFolder,
                    enable_checkpointing=True,
                    logger=[CSV_logger, tensorBoard_logger],
                    
                    #num_sanity_val_steps=0
                    
                    # logger=pl.loggers.TensorBoardLogger(save_dir="logs/"),
                    callbacks=[
                        checkpoint_callback, 
                        #DeviceStatsMonitor()
                    ],
                    accumulate_grad_batches=1,
                    precision="bf16-mixed"#"16-true"#"16-mixed"
                )
            
            else: 
                Globals.APP_LOGGER.info(f"Training Using GPU: {self._args.gpus}")
                
                trainer = pl.Trainer(
                    accelerator="gpu", 
                    devices=self._args.gpus,
                    max_epochs=self._args.epochs,
                    min_epochs=1, 
                    #profiler="simple", #profiler="advanced"
                    #default_root_dir= self._workingFolder,
                    enable_checkpointing=True,
                    logger=[CSV_logger, tensorBoard_logger],
                    
                    #num_sanity_val_steps=0
                    
                    # logger=pl.loggers.TensorBoardLogger(save_dir="logs/"),
                    callbacks=[
                        checkpoint_callback, 
                        #DeviceStatsMonitor()
                    ],
                    accumulate_grad_batches=1,
                    precision="bf16-mixed"#"16-true"#"16-mixed"
                )
            
        if self._device == torch.device("cpu"):
            trainer = pl.Trainer(
                accelerator="cpu",
                max_epochs=self._args.epochs,
                min_epochs=1,
                #profiler="simple", #profiler="advanced"
                #default_root_dir= self._workingFolder,
                enable_checkpointing=True,
                logger=[CSV_logger, tensorBoard_logger],
                callbacks=[
                    checkpoint_callback,
                    #DeviceStatsMonitor()
                ],
                accumulate_grad_batches=1,
                #precision="bf16-mixed"#"16-true"#"16-mixed"
            )
        
        if self._args.compile == 1:
            self._model = torch.compile(self._model)
        
        
        # trainer.fit(
        #     model=self._model,#torch.compile(traiableModel), 
        #     train_dataloaders=train_dataloader,
        #     val_dataloaders=validation_dataloader,
        #     ckpt_path=self._args.ckpt_path
        # )
        
        trainer.fit(
            model=self._model,#torch.compile(traiableModel), 
            datamodule=datamodule,
            ckpt_path=self._args.ckpt_path
        )
        
        
    
    def lightTestNetwork(self, testDataset: Dataset):

        

        test_dataloader = DataLoader(
            dataset = testDataset,     
            batch_size = self._args.batch_size, 
            shuffle = False, 
            num_workers = self._args.workers, 
            #worker_init_fn=trainingDataset.worker_init_fn,
            pin_memory=True,
            persistent_workers=True
        )

        
        trainer = pl.Trainer(accelerator="gpu", devices=[0])
        predictions = trainer.predict(model=self._model, dataloaders=test_dataloader, ckpt_path=self._args.ckpt_path)
    
        print(predictions)
        
            

    def trainNetwork(self, trainingDataset: Dataset, testDataset: Dataset, epochs: int = 1, batchSize: int = 1, workers: int = 0, lr: float = 1e-4, gamma: float = 0.99, startFactor: int = 1.0, endFactor: int = 1.0 ,logger = None, stepsForCheckpoint: int = -1, checkpointPath: str = None) -> np.array:
        
        assert self.__ModelWeights_Output_File != None and self.__ModelWeights_Output_File != "", "model_Output_File must be provided"
        
        if stepsForCheckpoint >= 0 and (checkpointPath == None or checkpointPath == ""):
            raise ValueError("checkpointPath must be provided if stepsForCheckpoint is greater than or equal to 0")
        
        if logger is None:
            logger = logging.getLogger('training')
        
        if workers > 0:
            torch.multiprocessing.set_start_method('spawn')
            #torch.multiprocessing.set_start_method('fork')

        if torch.cuda.is_available():
            torch.cuda.manual_seed(NetworkManager.__trainingRandomSeed)
            torch.cuda.manual_seed_all(NetworkManager.__trainingRandomSeed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        torch.manual_seed(NetworkManager.__trainingRandomSeed)
        
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=startFactor, end_factor=endFactor, total_iters=epochs)
        
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        train_dataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True, num_workers = workers, worker_init_fn=trainingDataset.worker_init_fn)
        test_dataloader  = DataLoader(dataset = testDataset,     batch_size = batchSize, shuffle = True, num_workers = workers, worker_init_fn=trainingDataset.worker_init_fn)

        logger.info(f"{'-'*40}Training{'-'*40}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batchSize}")
        logger.info(f"Workers: {workers}")
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Scheduler: {scheduler}")
        logger.info(f"Criterion: {criterion}")
        logger.info(f"Device: {self._device}")
        logger.info(f"Random Seed: {NetworkManager.__trainingRandomSeed}")
        logger.info(f"Model Name: {self._model.__class__.__name__}")
        logger.info(f"Model:\n{self.makeSummary()}")
        logger.info(f"{'-'*40}{'-'*40}")

       
        
        batchIter1 = len(range(int(len(train_dataloader))))
        batchIter2 = len(range(int(len(test_dataloader))))
        barInderval: float = 1/30
        descriptionLenght = 30

        avg_loss: float = 0.0
        accuracy: float = 0.0
        
        
        for epoch in range(epochs):

            epochsBar = tqdm.tqdm(total = epochs, initial=epoch+1,  colour="green", dynamic_ncols=True, ascii=True, mininterval=barInderval) #  dynamic_ncols=True
            batchBar = tqdm.tqdm(total=batchIter1, colour="green", dynamic_ncols=True, ascii=True, mininterval=barInderval)
            convBar = tqdm.tqdm(total=batchIter2,  colour="green", dynamic_ncols=True, ascii=True, mininterval=barInderval)
            
            epochsBar.set_description(f"Training Epoch {epoch+1}/{epochs}".ljust(descriptionLenght))
            #epochsBar.update()
            epochsBar.refresh()
            
            batchBar.set_description(f"Training: Batch 0/{batchIter1}".ljust(descriptionLenght))
            batchBar.reset()
            batchBar.refresh()

            convBar.set_description(f"Convalidation: Batch 0/{batchIter2}".ljust(descriptionLenght))
            convBar.reset()
            convBar.refresh()

            #======================TRAINING======================#
            
            # testDataset.closeAllConnections()
            # trainingDataset.closeAllConnections()
            
            #mposto il modella in modalità di addestramento
            self._model.train()  
            running_loss:float = 0.0
            
            
            for index, (data, labels) in enumerate(train_dataloader):

                batchBar.set_description(f"Training: Batch {index+1}/{batchIter1}".ljust(descriptionLenght))
                batchBar.update()
                batchBar.refresh()

                data = data.to(self._device)
                labels = labels.to(self._device)
                
                optimizer.zero_grad()                       # Azzerare i gradienti
                output = self._model.forward(data)         # Forward pass
                loss = criterion(output, labels)            # Calcolo della perdita
                
                # Backpropagation e aggiornamento dei pesi
                loss.backward()
                optimizer.step()

                # Accumulare la perdita per monitorare l'andamento
                running_loss += loss.item()
            
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
                
            #======================VALIDATION======================#

            total = 0
            correct = 0

            for index, (data, labels) in enumerate(test_dataloader):

                convBar.set_description(f"Convalidation: Batch {index+1}/{batchIter2}".ljust(descriptionLenght))
                convBar.update()
                convBar.refresh()


                data = data.to(self._device)
                labels = labels.to(self._device)
                labels = torch.argmax(labels, dim=1)

                outputs = self.predict(data, returnAsTensor= True)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
                    
            # Calcolare la perdita media per l'epoca
            avg_loss = running_loss / len(trainingDataset)
            
            epochsBar.close()
            batchBar.close()
            convBar.close()
            
            info = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}, Accuracy: {accuracy:.8f}, lr: {before_lr:.8f}"# -> {after_lr:.8f}"

            logger.info(info)
            print(f"{info}\n")
            terminal_width = shutil.get_terminal_size().columns
            print("-"*terminal_width)
            
            if (stepsForCheckpoint >= 0) and (epoch > 0 or stepsForCheckpoint == 0) and ((epoch + 1) % stepsForCheckpoint == 0):
                self._exportWeights(checkpointPath.replace('.pth',f'_epoch_{epoch + 1}.pth'))
        
        logger.info(f"{'-'*20}Training complete{'-'*20}")
    
    
    def predict(self, InputTensor:torch.Tensor, returnAsTensor = False):
        
        with torch.no_grad():
            self._model.eval()
            output = self._model(InputTensor.to(self._device))

        if returnAsTensor:
            return output

        all_probabilities = F.softmax(output, dim=1)
        predicted_class_index = torch.argmax(all_probabilities, dim=1).item()
        predicted_probability = all_probabilities[0, predicted_class_index].item()
        
        return predicted_class_index, predicted_probability, all_probabilities
    