import re
from sklearn.metrics import confusion_matrix
from torch import nn
import numpy as np


class _BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


# class MetricBase(_BaseObject):
#     pass

class MetricBase:
    def __init__(self, *args, **kwargs):
        pass