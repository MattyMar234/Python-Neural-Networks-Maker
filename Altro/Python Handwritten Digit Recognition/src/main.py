from matplotlib import transforms
import pygame
from Table import Table
from Network import Network
import torch
import time

from torchvision import datasets, transforms
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_probs = np.zeros((10),dtype=np.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
])

Model: Network = Network(1, 10).to(device)
Model.load_state_dict(torch.load("LaNet5_model.pth"))
Model.eval()

pygame.init()

# Dimensioni finestra
WINDOW_WIDTH = 600*2
WINDOW_HEIGHT = 330*3
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GREEN = (0, 255,0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

# Font
FONT = pygame.font.SysFont(None, 32)

# Configurazione finestra
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Neural Network Digit Recognition")


def draw_predictions(predictions: np.ndarray):
    """Mostra le probabilità di ciascuna classe"""
    #print(predictions)
    value_sum = predictions.sum()
    
    if value_sum == 0:
        for i, prob in enumerate(predictions):
            text = FONT.render(f"{i}: {prob:.2f}", True, BLUE)
            screen.blit(text, (1100, 60 + i * 36))
    else:
        argMax = predictions.argmax()
        
        for i, prob in enumerate(predictions):
            text = FONT.render(f"{i}: {prob:.2f}", True, GREEN if argMax == i else BLUE)
            screen.blit(text, (1100, 60 + i * 36))

def main():
    
    global last_probs
    
    running = True
    
    appState = {
        "drawing" : False,
        "erasing" : False,
        "lastUpdate" : 0.0
    }
    
    clock = pygame.time.Clock()
    
    
    tableSize = Table.getTableSize()
    grid = Table(int(WINDOW_WIDTH/2 - tableSize[0]/2), int(WINDOW_HEIGHT/2 - tableSize[1]/2))    

    while running:
        
        clock.tick(100)  # Aggiorna a 60 FPS

        screen.fill(BLACK)
        grid.draw(screen)
        draw_predictions(last_probs)
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    appState["drawing"] = True
                elif event.button == 3:
                    appState["erasing"] = True


            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    appState["drawing"] = False
                    appState["lastUpdate"] = time.time()
                elif event.button == 3:
                    appState["erasing"] = False
                    appState["lastUpdate"] = time.time()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid.clearTable()

        
        
        grid.update(appState)
        
        if grid.isChanged() and not appState["drawing"] and not appState["erasing"]:
            if time.time() - appState["lastUpdate"] >= 1:
                grid.clearIsChanged()
                tensor = torch.tensor(grid.getContent(), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
                
                tensor = tensor.to(device=device)
                probs = Model.predict(tensor)
                array = probs.cpu().numpy()[0]
                last_probs = array
                # print(probs)
                # print(torch.argmax(probs))
        
                
            
            
        
          
           

        # predictions = get_predictions()
        # draw_predictions(predictions)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
