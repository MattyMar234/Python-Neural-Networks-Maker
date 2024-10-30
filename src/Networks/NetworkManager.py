from dataclasses import dataclass
from torchvision import transforms
from torch import nn
import lightning as L
import torch
from torchmetrics import Metric
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchinfo import summary #conda install conda-forge::torchinfo -y

from .TrainingModel import *

from PIL import Image
import logging
from tqdm import tqdm
import os
import tqdm
import numpy as np
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor


from pytorch_lightning.loggers import TensorBoardLogger



class NetworkManager:

    __trainingRandomSeed: int = 42

    def __init__(self, device: torch.device, model: nn.Module, workingFolder: str, ModelWeights_Output_File:str = None, ModelWeights_Input_File: str = None, ckpt_file: str = None):
        
        self.__device: torch.device = device
        self.__model:nn.Module = model
        self._workingFolder:str = workingFolder
        
        if ModelWeights_Input_File == None or ModelWeights_Input_File == "":
            self.__ModelWeights_Input_File:str | None = None
        else:
            self.__ModelWeights_Input_File:str = os.path.join(workingFolder, ModelWeights_Input_File)
        
        self.__ModelWeights_Output_File:str = os.path.join(workingFolder, ModelWeights_Output_File)
        self._ckpt_file: str = os.path.join(workingFolder, ckpt_file)
        
        #self.__useLightModule: bool = useLightModule
        #self.__model_input_size = model_input_size
        
        if not os.path.exists(workingFolder):
            os.makedirs(workingFolder)
            
        # with open(self._ckpt_file, 'w') as f:
        #     pass
        
        self.__model.to(self.__device)
        
        if self.__ModelWeights_Input_File != None and self.__ModelWeights_Input_File != "":
            self._importWeights(path=self.__ModelWeights_Input_File)
    
    @property
    def parameters(self):
        return self.__model.parameters()
    
    @property
    def modelInfo(self):
        return self.__model.__str__()
    
   
    def makeSummary(self) -> str:
        colName = ['input_size', 'output_size', 'num_params', 'trainable']
        temp = summary(self.__model, input_size=self.__model.requestedInputSize(), col_width=20, col_names=colName, row_settings=['var_names'], verbose=0,depth=20)
        return temp.__repr__()
    

    def _exportWeights(self, path:str):
        torch.save(self.__model.state_dict(), path)
        logging.info(f'Modello salvato in {path}')
    

    def _importWeights(self, path:str):
        if os.path.isfile(path):
            self.__model.load_state_dict(torch.load(path))
            logging.info(f'Modello {path} importato con successo')
        else:
            logging.error(f'File {path} non trovato')
            raise FileNotFoundError(f'File {path} non trovato')


    def lightTrainNetwork(self, traiableModel: TrainableModel, trainingDataset: Dataset, testDataset: Dataset,
                          epochs: int = 1, batchSize: int = 1, workers: int = 0):
        
        
        train_dataloader = DataLoader(dataset = trainingDataset, batch_size = batchSize, shuffle = True, num_workers = workers, worker_init_fn=trainingDataset.worker_init_fn)
        test_dataloader  = DataLoader(dataset = testDataset,     batch_size = batchSize, shuffle = False, num_workers = workers, worker_init_fn=trainingDataset.worker_init_fn)
        
        L.seed_everything(NetworkManager.__trainingRandomSeed, workers= workers > 0)
        
        #profiler="advanced"
        logger = TensorBoardLogger("tb_logs", name="my_model")
        
        trainer = L.Trainer(
            accelerator="gpu", 
            devices=[0],
            max_epochs=epochs,
            min_epochs=1, 
            profiler="simple",
            default_root_dir= self._workingFolder,
            callbacks=[DeviceStatsMonitor()],
            enable_checkpointing=True,
            logger=logger
            
            #num_sanity_val_steps=0
            
            # logger=pl.loggers.TensorBoardLogger(save_dir="logs/"),
            # callbacks=[pl.callbacks.ModelCheckpoint(dirpath="checkpoints/", monitor="train_loss", mode="min", save_top_k=1)]
        )
        trainer.fit(
            model=traiableModel,#torch.compile(traiableModel), 
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
            #ckpt_path=self._ckpt_file
        )
        
        
    
    def lightTestNetwork(self, testDataset: Dataset, batchSize: int = 1, workers: int = 0):

        test_dataloader  = DataLoader(dataset = testDataset, batch_size = batchSize, shuffle = True, num_workers = workers, worker_init_fn=testDataset.worker_init_fn)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=1,
            min_epochs=1,
        )
        trainer.test(
            model=self.__model,
            dataloaders=test_dataloader,
            ckpt_path=self._ckpt_file
        )
            

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
        
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
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
        logger.info(f"Device: {self.__device}")
        logger.info(f"Random Seed: {NetworkManager.__trainingRandomSeed}")
        logger.info(f"Model Name: {self.__model.__class__.__name__}")
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
            self.__model.train()  
            running_loss:float = 0.0
            
            
            for index, (data, labels) in enumerate(train_dataloader):

                batchBar.set_description(f"Training: Batch {index+1}/{batchIter1}".ljust(descriptionLenght))
                batchBar.update()
                batchBar.refresh()

                data = data.to(self.__device)
                labels = labels.to(self.__device)
                
                optimizer.zero_grad()                       # Azzerare i gradienti
                output = self.__model.forward(data)         # Forward pass
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


                data = data.to(self.__device)
                labels = labels.to(self.__device)
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
            self.__model.eval()
            output = self.__model(InputTensor.to(self.__device))

        if returnAsTensor:
            return output

        all_probabilities = F.softmax(output, dim=1)
        predicted_class_index = torch.argmax(all_probabilities, dim=1).item()
        predicted_probability = all_probabilities[0, predicted_class_index].item()
        
        return predicted_class_index, predicted_probability, all_probabilities

    