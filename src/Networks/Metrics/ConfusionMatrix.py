import io
from typing import Dict, List, Tuple
from PIL import Image, ImageFile

from sklearn.metrics import confusion_matrix
from torch import nn
import numpy as np
import torch

from .MetricBase import MetricBase
import seaborn as sns
import matplotlib.pyplot as plt


class ConfusionMatrix(MetricBase):

    def __init__(self, *, classes_number: int, ignore_class: List[int] | None=None, mapFuntion: object | None = None, **kwargs):
        super().__init__(**kwargs)
        
       
        self._ignore_class: List[int] = ignore_class  # the class index to be removed
        self._classes_count: int = classes_number
        self._mapFuntion: object | None = mapFuntion
        self._class_labels: list = []
    
        
        for i in range(self._classes_count):
            if self._ignore_class is not None and i in self._ignore_class:
                continue
            if self._mapFuntion is not None:
                self._class_labels.append(self._mapFuntion(i))
            else:
                self._class_labels.append(i)
        
        
        self._matrix = np.zeros((self._classes_count, self._classes_count))
        
        
        # if ignore_class is None:
        #    self._matrix = np.zeros((self._classes_count, self._classes_count))
        # else:
        #     self._matrix = np.zeros((self._classes_count - len(ignore_class)), self._classes_count - (len(ignore_class)))
        

    def get_labels(self):
        if self._ignore_class is not None:
            return np.delete(self.labels, self._ignore_class)
        return self._classes_count
    
    def reset(self) -> None:
        self._matrix = np.zeros_like(self._matrix)

    def update(self, y_pr: torch.Tensor | np.ndarray, y_tr: torch.Tensor | np.ndarray) -> None:
        
        # assert isinstance(y_pr, torch.Tensor) or isinstance(y_pr, np.ndarray)
        # assert isinstance(y_tr, torch.Tensor) or isinstance(y_tr, np.ndarray)
        # #assert y_pr.shape == y_tr.shape and len(y_pr.shape) == 1 and len(y_tr.shape) == 1       
        
        # # Verifica che abbiano la stessa forma (batch_size,)
        # assert y_pr.shape == y_tr.shape
        
        # if not isinstance(y_pr, torch.Tensor):
        #     y_pr = torch.from_numpy(y_tr)
        
        # if not isinstance(y_tr, torch.Tensor):
        #     y_tr = torch.from_numpy(y_tr)
        
        y_tr = y_tr.flatten().to(torch.int64)
        y_pr = y_pr.flatten().to(torch.int64)
        
        # # Aggiorna la matrice di confusione ignorando le classi specificate
        # cm = np.zeros((self._classes_count, self._classes_count))
        # for true_class, pred_class in zip(y_tr, y_pr):
        #     #if self._ignore_class is None or (true_class not in self._ignore_class and pred_class not in self._ignore_class):
        #     cm[true_class, pred_class] += 1
        
        # Calcola la matrice di confusione usando PyTorch
        indices = y_tr * self._classes_count + y_pr
        cm = torch.bincount(indices, minlength=self._classes_count**2)
        cm = cm.reshape(self._classes_count, self._classes_count).cpu().numpy()
        
        self._matrix += cm
        
        

    def compute(self, figsize=(28, 16), cmap="viridis", showGraph: bool = False) -> None | Tuple[torch.Tensor, Dict[str, any]]:

        # To format the matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        # confusion_matrix(y_true, y_pred)
        # array([[2, 0, 0],  # two zeros were predicted as zeros
        #        [0, 0, 1],  # one 1 was predicted as 2
        #        [1, 0, 2]])  # two 2s were predicted as 2, and one 2 was 0
        matrix = self._matrix
        
        
        if self._ignore_class is not None:
            matrix = np.delete(matrix, self._ignore_class, axis=0)  # Rimuovi righe
            matrix = np.delete(matrix, self._ignore_class, axis=1)  # Rimuovi colonne
        
        
        
        
        #row_sums = np.sum(matrix, axis=1, keepdims=True)
        # cm_percent = matrix / np.sum(matrix, axis=1, keepdims=True) * 100
        # cm_percent = np.nan_to_num(cm_percent)  # Evita NaN divisi per 0
        
        row_sums = np.sum(matrix, axis=1, keepdims=True) + 1e-12
        cm_percent = (matrix / row_sums) * 100
        
        # Formatta le etichette con i conteggi assoluti e le percentuali
        group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]
        group_percentages = ["{0:.2f}%".format(value) for value in cm_percent.flatten()]
        
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(matrix.shape)
        
        plt.figure(figsize=figsize)
        
        
        graphData: Dict[str, any] = {
            "data":cm_percent, 
            "annot":labels, 
            "fmt":"", 
            "cmap":cmap, 
            "cbar":True,
            "xticklabels":self._class_labels, 
            "yticklabels":self._class_labels
        }
        
        
        sns.heatmap(
            data=cm_percent, 
            annot=labels, 
            fmt="", 
            cmap=cmap, 
            cbar=True,
            xticklabels=self._class_labels, 
            yticklabels=self._class_labels
        )
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        if showGraph:
            plt.show()
            plt.close()
            return None
        
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()  # Chiudi il grafico per liberare memoria

        # Ritorna all'inizio del buffer per leggerlo
        buf.seek(0)

        # Carica l'immagine dal buffer (opzionale: per operazioni successive)
        image: ImageFile = Image.open(buf)
        image = np.array(image)
        image_tensor = torch.tensor(image).permute(2, 0, 1) / 255.0
        
        buf.close()
        
        return image_tensor, graphData
        
        # return

        # results_vec = {"labels": self.get_labels(), "confusion matrix": self._matrix}

        # total = np.sum(self._matrix)
        # true_positive = np.diag(self._matrix)
        # sum_rows = np.sum(self._matrix, axis=0)
        # sum_cols = np.sum(self._matrix, axis=1)
        # false_positive = sum_rows - true_positive
        # false_negative = sum_cols - true_positive
        
        # # calculate accuracy
        # overall_accuracy = np.sum(true_positive) / total
        # results_scalar = {"OA": overall_accuracy}

        # # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        # p0 = np.sum(true_positive) / total
        # pc = np.sum(sum_rows * sum_cols) / total ** 2
        # kappa = (p0 - pc) / (1 - pc)
        # results_scalar["Kappa"] = kappa

        # # Per class recall, prec and F1
        # recall = true_positive / (sum_cols + 1e-12)
        # precision = true_positive / (sum_rows + 1e-12)
        # f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

        # results_vec["R"] = recall
        # results_vec["P"] = precision
        # results_vec["F1"] = f1
        
        
        # # Just in case we get a division by 0, ignore/hide the error
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     iou = true_positive / (true_positive + false_positive + false_negative)
        #     results_vec["IoU"] = iou
        #     results_scalar["mIoU"] = np.nanmean(iou)

        # # Per class accuracy
        # cl_acc = true_positive / (sum_cols + 1e-12)
        # results_vec["Acc"] = cl_acc

        # # weighted measures
        # prob_c = sum_rows / total
        # prob_r = sum_cols / total
        # recall_weighted = np.inner(recall, prob_r)
        # results_scalar["wR"] = recall_weighted
        # precision_weighted = np.inner(precision, prob_r)
        # results_scalar["wP"] = precision_weighted
        # f1_weighted = 2 * (recall_weighted * precision_weighted) / (recall_weighted + precision_weighted)
        # results_scalar["wF1"] = f1_weighted
        # random_accuracy = np.inner(prob_c, prob_r)
        # results_scalar["RAcc"] = random_accuracy


        # return results_vec, results_scalar

    @staticmethod
    def showConfusionMatrix(confusionMatrix, class_labels=None, figsize=(10, 8), cmap="Blues", annot=True, fmt=".2f") -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        matrix = confusionMatrix.compute()

        plt.figure(figsize=figsize)
        sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, cbar=True,
                    xticklabels=class_labels, yticklabels=class_labels)
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        # plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix")
        # plt.colorbar()
        # tick_marks = np.arange(len(self.get_labels()))
        # plt.xticks(tick_marks, self.get_labels(), rotation=45)
        # plt.yticks(tick_marks, self.get_labels())
        # plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()