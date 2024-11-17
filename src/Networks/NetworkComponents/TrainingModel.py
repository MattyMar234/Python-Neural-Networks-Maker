from Globals import *
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import lightning as L
from torchinfo import summary
from abc import ABC, abstractmethod
from typing import Dict, Final, Protocol
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from DatasetComponents.DataModule.DataModuleBase import DataModuleBase
import Globals
from Networks.Metrics.ConfusionMatrix import ConfusionMatrix
from .NeuralNetworkBase import LightModelBase, ModelBase
import torchmetrics
from enum import Enum, auto


class ShedulerType(Enum):

    LINEAR = "linearLR"
    EXP = "expLR"
    STEP = "stepLR"
    COSINE = "cosine"
    COSINE_RESTARTS = "cosineWR"
    CONSTANT = "constantLR"
    POLY = "polyLR"
    MULTI_STEP = "multistepLR"
    #MULTI_STEP_DECAY = "multi_step_decay"
    # ONE_CYCLE = "one_cycle"
    # ONE_CYCLE_DECAY = "one_cycle_decay"
    NONE = "none"
    
    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())



class TraingBase(LightModelBase):
    
    AVG_TRAINING_LOSS_LABEL_NAME: Final[str] = "avg_train_loss"
    AVG_VALIDATION_LOSS_LABEL_NAME: Final[str] = "avg_val_loss"
    VAL_ACCURACY_LABEL_NAME: Final[str] = "val_accuracy"
    LR_LABEL_NAME: Final[str] = "learning_rate"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        datamodule: DataModuleBase = kwargs.get("datamodule")
        self._kwargs = kwargs
        self._learning_rate: float = kwargs['lr']
        
        if self._learning_rate is None or (self._learning_rate > 1 or self._learning_rate < 0):
            self._learning_rate = 1e-3
        
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=['net'])
        
        
        self._lossFunction = self.configure_loss()
        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_loss_metric = torchmetrics.MeanMetric()
        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self._output_Classes)
        #self.confusion_matrix_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self._output_Classes)
        
        
        self.confusion_matrix_metric = ConfusionMatrix(classes_number=self._output_Classes, ignore_class= datamodule.classesToIgnore(), mapFuntion=datamodule.map_classes)
    
        self._last_avg_trainLoss: float = -1.0
        self._last_avg_valLoss: float = -1.0
    
    
    def _make_sheduler(self, optimizer) -> Dict[str, any]:
        
        scheduler: Dict[str, any] = {}
        shedulerType: ShedulerType = ShedulerType(self._kwargs[Globals.SCHEDULER_TYPE])
        scheduler['interval'] = self._kwargs[Globals.SCHEDULER_STEP_TYPE]
        
        match shedulerType:
            case ShedulerType.NONE:
                scheduler['scheduler'] = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self._kwargs[START_FACTOR], end_factor=self._kwargs[END_FACTOR], total_iters=self._kwargs[EPOCHS])
                
            case ShedulerType.LINEAR:
                scheduler['scheduler'] = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self._kwargs[START_FACTOR], end_factor=self._kwargs[END_FACTOR], total_iters=self._kwargs[EPOCHS])
            
            case ShedulerType.EXP:
                scheduler['scheduler'] = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self._kwargs[SCHEDULER_GAMMA])
            
            case ShedulerType.STEP:
                scheduler['scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self._kwargs[SCHEDULER_STEP_SIZE], gamma=self._kwargs[SCHEDULER_GAMMA])
            
            case ShedulerType.COSINE:
                scheduler['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._kwargs[SCHEDULER_STEP_SIZE], eta_min=self._kwargs[ETA_MIN])
            
            case ShedulerType.COSINE_RESTARTS:
                scheduler['scheduler'] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self._kwargs[SCHEDULER_STEP_SIZE], T_mult=self._kwargs[T_MULT], eta_min=self._kwargs[ETA_MIN])
            
            case ShedulerType.CONSTANT:
                scheduler['scheduler'] = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=self._kwargs[FACTOR], total_iters=self._kwargs[SCHEDULER_STEP_SIZE])
            
            case ShedulerType.POLY:
                scheduler['scheduler'] = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self._kwargs[SCHEDULER_STEP_SIZE], power=self._kwargs[POWER])
            
            case ShedulerType.MULTI_STEP:
                scheduler['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._kwargs[MILESTONES], gamma=self._kwargs[SCHEDULER_GAMMA])
            
            # case ShedulerType.MULTI_STEP_DECAY:
            #     scheduler['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self._kwargs[MILESTONES], gamma=self._kwargs[SCHEDULER_GAMMA])
            
            # case ShedulerType.ONE_CYCLE:
            #     scheduler['scheduler'] = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, total_steps=100)
            
            # case ShedulerType.ONE_CYCLE_DECAY:
            #     scheduler['scheduler'] = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, total_steps=100, anneal_strategy='linear')
            
                
            case _:
                raise ValueError(f"Invalid sheduler type: {shedulerType}")
        
        Globals.APP_LOGGER.info(f"Scheduler: {scheduler}")
        
        if self.trainer.current_epoch > 0:
            Globals.APP_LOGGER.info(f"update scheduler epoch")
            scheduler.last_epoch = self.trainer.current_epoch - 1
        
        
        return scheduler
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Modifica il checkpoint prima che venga caricato
        
        if 'optimizer_states' in checkpoint:
            #checkpoint.pop('optimizer_states', None)  # Rimuovi lo stato dell'optimizer

            optimizer_states = checkpoint['optimizer_states']
            
            # Supponiamo di voler modificare il learning rate nel stato dell'optimizer
            for optimizer_state in optimizer_states:
                for group in optimizer_state['param_groups']:
                    # Cambia il learning rate a 0.001 (o qualsiasi valore tu voglia)
                    group['lr'] = self._learning_rate

        return checkpoint
    
    # @abstractmethod 
    # def configure_optimizers(self) -> tuple[list, list]:
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
    #     return [optimizer], [self._make_sheduler()]
    
    @abstractmethod  
    def _commonStep(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        #y_hat = self.__net(x)
        y_hat = self.forward(x)
        loss = self._lossFunction(y_hat, y.squeeze(1))
        return {"loss": loss, "y_hat": y_hat}
    
    @abstractmethod
    def configure_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()


    @abstractmethod
    def configure_optimizers(self) -> tuple[list, list]:
        ...
        
    @abstractmethod
    def compute_accuracy_metric(self, values: dict[str, any], batch_x: torch.Tensor, batch_y: torch.Tensor) -> None :
        self.val_accuracy_metric(values['y_hat'], batch_y)
        
        
    #================================== STEPS ==================================#
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(x=batch[0], y=batch[1], batch_idx=batch_idx)
        
        #Accumolo il valore della loss
        self.train_loss_metric.update(values['loss'])
        
        #self.log_dict(values, on_step=True, on_epoch=False, prog_bar=True)
        return values
    
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(x=batch[0], y=batch[1], batch_idx=batch_idx)
        
        #Accumolo il valore della loss
        self.val_loss_metric.update(values['loss'])
        
        #calcolo l'accuratezza e accimolo il valore
        self.compute_accuracy_metric(values, batch_imgs, batch_labels)
        self.update_confusion_matrix(y_hat=values['y_hat'], y=batch_labels)
        

        return values
    
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._commonStep(x=batch[0], y=batch[1], batch_idx=batch_idx)
    
    @abstractmethod
    def update_confusion_matrix(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        self.confusion_matrix_metric.update(y_pr=y_hat, y_tr=y)


    @abstractmethod
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        outputs = self.__net(batch_imgs)
        y_hat = torch.argmax(outputs, dim=1)
        return y_hat
    
    #================================== EPOCHS ==================================#
    def on_train_epoch_end(self):
        
        '''Calcolo il valore della loss media della fase di training'''
        
        # self._last_avg_trainLoss = self.train_loss_metric.compute()
        # self.train_loss_metric.reset()
        pass
        

    def on_validation_epoch_end(self):
        
        '''Calcolo il valore della loss media della fase di validation e calcolo
            il valore dell'accuratezza. E in fine plotto i valori. 
        '''
        
        self._last_avg_trainLoss = self.train_loss_metric.compute()
        self._last_avg_valLoss = self.val_loss_metric.compute()
        val_acc = self.val_accuracy_metric.compute()
        image_tensor, _ = self.confusion_matrix_metric.compute()
        
        self.train_loss_metric.reset()
        self.val_loss_metric.reset()
        self.val_accuracy_metric.reset()
        self.confusion_matrix_metric.reset()
        
        

        t_dict = {
            TraingBase.AVG_TRAINING_LOSS_LABEL_NAME : self._last_avg_trainLoss,
            TraingBase.AVG_VALIDATION_LOSS_LABEL_NAME : self._last_avg_valLoss,
            TraingBase.VAL_ACCURACY_LABEL_NAME : val_acc,
            TraingBase.LR_LABEL_NAME: self.lr_schedulers().get_last_lr()[0]  # Ottieni il learning rate attuale
        }
        
        #if not self.sanity_checking:
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(f"Confusion Matrix epoch {self.current_epoch}", image_tensor, global_step=self.current_epoch)
            #self.save_epoch_metrics()
        
        self.log_dict(t_dict, on_epoch=True, prog_bar=True)
    
    def save_epoch_metrics(self, data: dict[str, any]):
        data['epoch'] = self.current_epoch


class ImageClassificationBase(TraingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
            
    # def configure_optimizers(self) -> tuple[list, list] :
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
    #         'interval': 'epoch',
    #     }
    #     return [optimizer], [scheduler]
    


class Semantic_ImageSegmentation_TrainingBase(TraingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    
    def configure_loss(self) -> nn.Module:
        
        if self._output_Classes == 1:
            return nn.BCELoss()
        else:
            #return nn.NLLLoss()
            print(f"CrossEntropyLoss loadded weights: {self._datamodule.getWeights}")
            return nn.CrossEntropyLoss(weight=self._datamodule.getWeights)
    
    
    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
        scheduler = self._make_sheduler(optimizer)
        
        
        # scheduler.last_epoch = self.current_epoch
        # scheduler.get_lr()  # Questo aggiorna internamente il learning rate del scheduler
        # new_lr = optimizer.param_groups[0]['lr']
        
        # # Aggiorna manualmente l'LR (se necessario)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lr  # Ricalcola e aggiorna l'LR
        
        Globals.APP_LOGGER.info(f"lr: {optimizer.param_groups[0]['lr']}")
        
        
        return [optimizer], [scheduler]
    
    def compute_accuracy_metric(self, values: dict[str, any], batch_x: torch.Tensor, batch_y: torch.Tensor) -> None :
        predicted_classes = torch.argmax(values['y_hat'], dim=1)   # shape: (1, 48, 48)
        target_classes = torch.argmax(batch_y, dim=1)             # shape: (1, 48, 48)
    
    
        
        #aggiunge i risultati di ogni batch alla metrica
        self.val_accuracy_metric.update(predicted_classes, target_classes)
     
    def update_confusion_matrix(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        
        #per eliminare la notazione in oneHot
        y_hat = torch.argmax(y_hat, dim=1)
        y = torch.argmax(y, dim=1)
        
        
        self.confusion_matrix_metric.update(y_pr=y_hat, y_tr=y)
     
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        y_hat = self.__net(batch[0])
        probabilities = F.softmax(y_hat, dim=1)  # Calcola le probabilità
        featureMap = torch.argmax(probabilities, dim=1)
        return featureMap
        



        
        
    