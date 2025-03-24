import torch.nn.functional as F  # Importa le funzioni di PyTorch
from torch import nn, optim 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch


class Network(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(Network, self).__init__()
        
        #Rete convoluzionale
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2),  # Primo strato convoluzionale
            nn.Tanh(),                                                         # Funzione di attivazione Tanh
            nn.AvgPool2d(kernel_size=2, stride=2),                             # Pooling medio
            nn.Conv2d(6, 16, kernel_size=5, stride=1),                         # Secondo strato convoluzionale
            nn.Tanh(),                                                         # Funzione di attivazione Tanh
            nn.AvgPool2d(kernel_size=2, stride=2),                             # Pooling medio
        )
        
        # Strati completamente connessi
        self.fc_net = nn.Sequential(
            nn.Flatten(),                         # Flatten dell'output della parte convoluzionale
            nn.Linear(16 * 5 * 5, 120),           # Primo strato completamente connesso
            nn.Tanh(),                            # Funzione di attivazione Tanh
            nn.Linear(120, 84),                   # Secondo strato completamente connesso
            nn.Tanh(),                            # Funzione di attivazione Tanh
            nn.Linear(84, num_classes)            # Strato di output
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x

    def predict(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)