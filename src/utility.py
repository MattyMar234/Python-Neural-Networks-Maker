import numpy as np
from torchvision import transforms
import torch


def processImage(self, frame_rgb):
        transform = transforms.Compose([
            #transforms.Grayscale(),  # Converti in bianco e nero
            transforms.Resize((28, 28)),  # Ridimensiona l'immagine a 28x28
            #transforms.Normalize((0.5,), (0.5,)),  # Normalizza i valori a [-1, 1]
            transforms.ToTensor()  # Converti in un tensore
        ])

        # Estrai i canali R, G, B
        r_channel = frame_rgb[:, :, 0].flatten()  # Canale rosso
        g_channel = frame_rgb[:, :, 1].flatten()  # Canale verde
        b_channel = frame_rgb[:, :, 2].flatten()  # Canale blu

        
        # Combina i canali in un array 1D
        combined_bytes = np.concatenate((r_channel, g_channel, b_channel))
        image: np.array = (combined_bytes.astype(np.float32) / 255.0)
        
        tensor = torch.from_numpy(image).to(self.__device)
        tensor = tensor.view(1, 28 * 28 * 3)

        output = self.__model(tensor)

        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0, predicted_class].item()
        
        return predicted_class, predicted_probability, probabilities