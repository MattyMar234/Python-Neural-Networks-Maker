import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import lightning as L
from torchinfo import summary
from abc import ABC, abstractmethod
from typing import Final, Protocol
import torchmetrics
from torchmetrics import Metric
from .NeuralNetworkBase import LightModelBase, ModelBase
import torchmetrics
from enum import Enum, auto



class TrainModel_Type(Enum):
    Predictions = auto()
    ImageClassification = auto()
    ObjectDetection = auto()
    Segmentation = auto()
    Other = auto()
    


class TraingBase(LightModelBase):
    
    AVG_TRAINING_LOSS_LABEL_NAME: Final[str] = "avg_train_loss"
    AVG_VALIDATION_LOSS_LABEL_NAME: Final[str] = "avg_val_loss"
    VAL_ACCURACY_LABEL_NAME: Final[str] = "val_accuracy"
    LR_LABEL_NAME: Final[str] = "learning_rate"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
    
        self._last_avg_trainLoss: float = -1.0
        self._last_avg_valLoss: float = -1.0
    
    
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
    def compute_accuracy_metric(self, values: dict, batch_imgs, batch_labels) -> None :
        self.val_accuracy_metric(values['y_hat'], batch_labels)
        
    #================================== STEPS ==================================#
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(batch_imgs, batch_labels, batch_idx)
        
        self.train_loss_metric.update(values['loss'])
        #self.log_dict(values, on_step=True, on_epoch=False, prog_bar=True)
        return values
    
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(batch_imgs, batch_labels, batch_idx)
        
        self.val_loss_metric.update(values['loss'])
        self.compute_accuracy_metric(values, batch_imgs, batch_labels.squeeze(1))
        return values
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._commonStep(batch[0], batch[1], batch_idx)


    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        outputs = self.__net(batch_imgs)
        preds = torch.argmax(outputs, dim=1)
        return preds
    
    #================================== EPOCHS ==================================#
    def on_train_epoch_end(self):
        
        self._last_avg_trainLoss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        
        
        t_dict = {
            TraingBase.AVG_TRAINING_LOSS_LABEL_NAME : self._last_avg_trainLoss,
        }
        
        self.log_dict(t_dict, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._last_avg_valLoss = self.val_loss_metric.compute()
        val_acc = self.val_accuracy_metric.compute()
        
        self.val_loss_metric.reset()
        self.val_accuracy_metric.reset()
        

        t_dict = {
            TraingBase.AVG_VALIDATION_LOSS_LABEL_NAME : self._last_avg_valLoss,
            TraingBase.VAL_ACCURACY_LABEL_NAME : val_acc,
            TraingBase.LR_LABEL_NAME: self.lr_schedulers().get_last_lr()[0]  # Ottieni il learning rate attuale
        }
        
        self.log_dict(t_dict, on_epoch=True, prog_bar=True)
    
    def save_epoch_metrics(self):
        pass


class ImageClassificationBase(TraingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
            
    def configure_optimizers(self) -> tuple[list, list] :
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]
    


class Semantic_ImageSegmentation_TrainingBase(TraingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]
    
    def configure_loss(self) -> nn.Module:
        
        if self._kwargs['output_Classes'] == 1:
            return nn.BCELoss()
        else:
            #return nn.NLLLoss()
            return nn.CrossEntropyLoss()
    
    
     
     
     
     
        



        
        
    