import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import lightning as L
from torchinfo import summary
from abc import ABC
from typing import Protocol
from torchmetrics import Metric
from .NetworkComponents.NeuralNetworkBase import ModelBase
import torchmetrics
from enum import Enum, auto

class TrainableModel(Protocol):
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        ...
        
    #STEPS

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        ...
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        ...

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        ...

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        ...
        
    #EPOCHS

    def on_train_epoch_end(self, outputs):
        ...
        
    def on_validation_epoch_end(self):
        ...
        
    def on_test_epoch_end(self):
        ...
        
    def on_train_epoch_start(self):
        ...
        
    def on_validation_epoch_start(self):
        ...
    
    def on_test_epoch_start(self):
        ...

class TrainModel_Type(Enum):
    Predictions = auto()
    ImageClassification = auto()
    ObjectDetection = auto()
    Segmentation = auto()
    Other = auto()
    


class _TraingBase(L.LightningModule):
    def __init__(self, net: nn.Module = None, lr: float = 1e-3):
        super().__init__()
        self._net = net
        self._lr = lr
        self._lossFunction = nn.CrossEntropyLoss()
        
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self._net.outputClasses)
        self.train_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self._net.outputClasses)
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self._net.outputClasses)
        self.val_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self._net.outputClasses)
       
        
    def _commonStep(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        y_hat = self._net(x)
        loss = self._lossFunction(y_hat, y)
        return {"loss": loss}


class _ImageClassificationBase(_TraingBase):
    def __init__(self, net: ModelBase = None, lr: float = 1e-2):
        super().__init__(net, lr)
        
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr, weight_decay=1e-3,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, verbose=True)
        return [optimizer], [scheduler]
    
    #================================== STEPS ==================================#
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        return self._commonStep(batch_imgs, batch_labels, batch_idx)
    
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        return self._commonStep(batch_imgs, batch_labels, batch_idx)
    
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        outputs = self._net(batch_imgs)
        preds = torch.argmax(outputs, dim=1)
        return preds
    
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        batch_imgs, batch_labels = batch
        return self._commonStep(batch_imgs, batch_labels, batch_idx)

    #================================== EPOCHS ==================================#
    # def on_train_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     return {"avg_loss": avg_loss}

    # def on_validation_epoch_end(self, outputs):
    #     pass

    # def on_test_epoch_end(self, outputs):
    #     pass
    
    # def on_train_epoch_start(self, outputs):
    #     pass
        
    # def on_validation_epoch_start(self, outputs):
    #     pass
    
    # def on_test_epoch_start(self, outputs):
    #     pass
    
    #================================== METRICS ==================================#

    def save_metrics(self):
        pass

class BaseTrainerModel_ImageClassification(_ImageClassificationBase):
    def __init__(self, net: nn.Module = None, lr: float = 1e-3):
        super().__init__(net, lr)
        


class LightTrainerModel_ImageClassification(_ImageClassificationBase):
    def __init__(self, net: nn.Module = None, lr: float = 1e-3):
        #L.LightningModule.__init__(self)
        _ImageClassificationBase.__init__(self, net, lr)
        
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        values = _ImageClassificationBase.training_step(self, batch, batch_idx)
        self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True)#,reduce_fx="mean")
        return values


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        values = _ImageClassificationBase.validation_step(self, batch, batch_idx)
        self.log_dict(values, on_step=True, on_epoch=True)#, batch_size=self.trainer.num_val_batches)
        return values

    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        values = _ImageClassificationBase.test_step(self, batch, batch_idx)
        self.log_dict(values, on_step=True, on_epoch=True, batch_size=self.trainer.datamodule.batch_size)
        return values

        
        
        
    