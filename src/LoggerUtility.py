
import re
import json
import logging
import os
from typing import Optional
import Globals

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


class ColorCodes:
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"

COLORS = {
    'DEBUG': '\033[94m',    # Blu
    'INFO': Fore.GREEN,     # Verde
    'WARNING': '\033[93m',  # Giallo
    'ERROR': '\033[91m',    # Rosso
    'CRITICAL': '\033[95m', # Magenta
    'RESET': '\033[0m'      # Resetta il colore
}

class ColoredFormatter(logging.Formatter):
    
    def __init__(self, base_formatter):
        """
        Inizializza con un formatter esistente.
        """
        self.base_formatter = base_formatter
    
    def format(self, record):
        """
        Colora solo il livello di log.
        """
        # Colore basato sul livello
        color = COLORS.get(record.levelname, COLORS['RESET'])
        levelname_colored = f"[{color}{record.levelname}{COLORS['RESET']}]"
        
        # Salva il livello originale e lo sostituisce con quello colorato
        original_levelname = record.levelname
        record.levelname = levelname_colored
        
        # Usa il formatter base per generare il messaggio
        message = self.base_formatter.format(record)
        
        # Ripristina il livello originale per evitare effetti collaterali
        record.levelname = original_levelname
        return message

def makeSeparetor(ch: str, label: Optional[str] = None) -> str:
    assert len(ch) == 1, "ch must be 1 character long"
    size = os.get_terminal_size()[0] - 12
    
    if label is None:
        return size * ch
    else:
        return label.center(size, ch)

def _get_handler_by_name(logger: logging.Logger, name: str) -> logging.Handler:
    if Globals.PYTHON_VERSION >= 3.12:  # Controllo sulla versione di Python
        return logging.getHandlerByName(name)
    else:
        # Metodo alternativo per versioni precedenti
        for handler in logger.handlers:
            if handler.get_name() == name:
                return handler
    return None


def setupLogging() -> None:
    
    #global APP_LOGGER
    
    if not os.path.exists(Globals.LOGS_FOLDER):
        os.makedirs(Globals.LOGS_FOLDER)
    
    with open(Globals.LOGGERS_CONFIG_FILE, 'r') as file:
        configuration = json.load(file)
    
    
    logging.config.dictConfig(configuration)
    
    Globals.APP_LOGGER = logging.getLogger(Globals.APP_LOGGER_NAME)
    consoleHandler: logging.Handler = _get_handler_by_name(logger=Globals.APP_LOGGER, name=Globals.CONSOLE_LOGGER_NAME)
    
    console_format = consoleHandler.formatter
    consoleHandler.setFormatter(ColoredFormatter(console_format))
    
    Globals.APP_LOGGER.info("Logger setup completed.")
