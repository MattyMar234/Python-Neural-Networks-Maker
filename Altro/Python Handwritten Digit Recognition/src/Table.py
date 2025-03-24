from typing import Final, Tuple
import pygame
import numpy as np


class Table:
    
    _COLUMS: Final[int] = 28
    _ROWS: Final[int] = 28
    _CELL_SIZE = 32
    _EMPTY_COLOR = (40, 40, 40)
    _FILL_COLOR = (255, 255, 255)
    _LINE_COLOR = (200, 200, 200)
    _BORDER_WIDTH = 4
    
    _WIDTH = _COLUMS*_CELL_SIZE
    _HEIGHT = _ROWS*_CELL_SIZE
   
    @staticmethod
    def getTableSize() -> Tuple[int, int] :
        return (Table._WIDTH, Table._HEIGHT)
    
    def __init__(self, pos_x: int = 0, pos_y: int = 0) -> None:
        
        self._posX = pos_x
        self._posY = pos_y
        self._grid = np.zeros((Table._ROWS, Table._COLUMS), dtype=np.uint8)
        self._isChanged: bool = False
   
    def getContent(self) -> np.ndarray:
        return self._grid.copy()
        
    def clearTable(self) -> None:
        self._grid = np.zeros((Table._ROWS, Table._COLUMS), dtype=np.uint8)
        
    def isChanged(self) -> bool :
        return self._isChanged
    
    def clearIsChanged(self) -> None:
        self._isChanged = False
    
    def update(self, appState: dict) -> None:
        
        if not (appState["erasing"] or appState["drawing"]):
            return
        
        value = 0 if appState["erasing"] else 1 
        mouse_x, mouse_y = pygame.mouse.get_pos()
        BRUSH_RADIUS = 1
        
        if mouse_x > self._posX + Table._WIDTH or mouse_x < self._posX:
            return
        if mouse_y > self._posY + Table._HEIGHT or mouse_y < self._posY:
            return
        
        col = ((mouse_x - self._posX) // Table._CELL_SIZE)
        row = ((mouse_y - self._posY) // Table._CELL_SIZE)
        
        if 0 <= row < Table._ROWS and 0 <= col < Table._COLUMS:
            for r in range(max(0, row - BRUSH_RADIUS), min(Table._ROWS, row + BRUSH_RADIUS + 1)):
                for c in range(max(0, col - BRUSH_RADIUS), min(Table._COLUMS, col + BRUSH_RADIUS + 1)):
                    if (r - row) ** 2 + (c - col) ** 2 <= BRUSH_RADIUS ** 2:
                        self._grid[r, c] = value
            
            self._isChanged = True
    
    
    def draw(self, screen) -> None :
        for row in range(Table._ROWS):
            for col in range(Table._COLUMS):
                
                rect = pygame.Rect(self._posX + col*Table._CELL_SIZE , self._posY + row*Table._CELL_SIZE, Table._CELL_SIZE, Table._CELL_SIZE)
                color = Table._FILL_COLOR if self._grid[row, col] == 1 else Table._EMPTY_COLOR
                pygame.draw.rect(screen, color, rect)
        
        
        for row in range(Table._ROWS + 1):
            if row == 0 or row == (Table._ROWS):
                pygame.draw.line(screen, Table._LINE_COLOR, (self._posX, row*Table._CELL_SIZE + self._posY), (self._posX + Table._COLUMS*Table._CELL_SIZE, row*Table._CELL_SIZE + self._posY), width=Table._BORDER_WIDTH)
            else:
                pygame.draw.line(screen, Table._LINE_COLOR, (self._posX, row*Table._CELL_SIZE + self._posY), (self._posX + Table._COLUMS*Table._CELL_SIZE, row*Table._CELL_SIZE + self._posY), width=1)  
              
        for col in range(Table._COLUMS + 1):
            if col == 0 or col == (Table._COLUMS):
                pygame.draw.line(screen, Table._LINE_COLOR, (self._posX + col*Table._CELL_SIZE, self._posY), (self._posX + col*Table._CELL_SIZE, Table._ROWS*Table._CELL_SIZE + self._posY), width=Table._BORDER_WIDTH)
            else:
                pygame.draw.line(screen, Table._LINE_COLOR, (self._posX + col*Table._CELL_SIZE, self._posY), (self._posX + col*Table._CELL_SIZE, Table._ROWS*Table._CELL_SIZE + self._posY), width=1)