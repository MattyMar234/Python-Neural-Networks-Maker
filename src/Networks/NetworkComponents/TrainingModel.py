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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._learning_rate: float = kwargs['lr']
        
        if self._learning_rate is None or (self._learning_rate > 1 or self._learning_rate < 0):
            self._learning_rate = 1e-3
        
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=['net'])
        
        
        self._lossFunction = nn.CrossEntropyLoss()
        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_loss_metric = torchmetrics.MeanMetric()
        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self._output_Classes)
        
    def _commonStep(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        #y_hat = self.__net(x)
        y_hat = self.forward(x)
        loss = self._lossFunction(y_hat, y)
        return {"loss": loss, "y_hat": y_hat}


    @abstractmethod
    def configure_optimizers(self) -> any:
        ...


class ImageClassificationBase(TraingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]
    
    #================================== STEPS ==================================#
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(batch_imgs, batch_labels, batch_idx)
        
        self.train_loss_metric.update(values['loss'])
        #self.log_dict(values, on_step=True, on_epoch=False, prog_bar=True)
        return values
    
 
    
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        values = self._commonStep(batch_imgs, batch_labels, batch_idx)
        
        self.val_loss_metric.update(values['loss'])
        self.val_accuracy_metric(values['y_hat'], batch_labels)
        
        return values
    
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        outputs = self.__net(batch_imgs)
        preds = torch.argmax(outputs, dim=1)
        return preds
    
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        return self._commonStep(batch_imgs, batch_labels, batch_idx)

    #================================== EPOCHS ==================================#
    def on_train_epoch_end(self):
        
        avg_train_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        
        
        t_dict = {
            TraingBase.AVG_TRAINING_LOSS_LABEL_NAME : avg_train_loss,
        }
        
        self.log_dict(t_dict, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_metric.compute()
        val_acc = self.val_accuracy_metric.compute()
        
        self.val_loss_metric.reset()
        self.val_accuracy_metric.reset()
        

        t_dict = {
            TraingBase.AVG_VALIDATION_LOSS_LABEL_NAME : avg_val_loss,
            TraingBase.VAL_ACCURACY_LABEL_NAME : val_acc,
            'learning_rate': self.lr_schedulers().get_last_lr()[0]  # Ottieni il learning rate attuale
        }
        
        self.log_dict(t_dict, on_epoch=True, prog_bar=True)
        
        
    
   

    # def on_test_epoch_end(self, outputs):
    #     pass
    
    # def on_train_epoch_start(self, outputs):
    #     pass
        
    # def on_validation_epoch_start(self, outputs):
    #     pass
    
    # def on_test_epoch_start(self, outputs):
    #     pass
    
    #================================== METRICS ==================================#

    def save_epoch_metrics(self):
        pass

class ImageSegmentation_TrainingBase(ImageClassificationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)#, weight_decay=1e-3,)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._commonStep(batch[0], batch[1], dataloader_idx)
     
     
     
     
        


# class LightTrainerModel_ImageClassification(_ImageClassificationBase):
#     def __init__(self, net: nn.Module = None, lr: float = 1e-3):
#         #L.LightningModule.__init__(self)
#         _ImageClassificationBase.__init__(self, net, lr)
        
#     # def training_step(self, batch: torch.Tensor, batch_idx: int):
#     #     values = _ImageClassificationBase.training_step(self, batch, batch_idx)
#     #     self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True)#,reduce_fx="mean")
#     #     return values


#     # def validation_step(self, batch: torch.Tensor, batch_idx: int):
#     #     values = _ImageClassificationBase.validation_step(self, batch, batch_idx)
#     #     self.log_dict(values, on_step=True, on_epoch=True)#, batch_size=self.trainer.num_val_batches)
#     #     return values

    
#     def test_step(self, batch: torch.Tensor, batch_idx: int):
#         values = _ImageClassificationBase.test_step(self, batch, batch_idx)
#         self.log_dict(values, on_step=True, on_epoch=True, batch_size=self.trainer.datamodule.batch_size)
#         return values

        
        
        
    