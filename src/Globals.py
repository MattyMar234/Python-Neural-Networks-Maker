from logging import Logger
import os
from typing import Final, List
from pathlib import Path

DATA_PATH: Final[Path] = Path(os.getcwd()).parent.absolute()
MODELS_OUTPUT_FOLDER = os.path.join(DATA_PATH, 'Models')


#============================= args key =============================#
EPOCHS: Final[str] = 'epochs'

LEARNING_RATE: Final[str] = 'lr'
SCHEDULER_TYPE: Final[str] = 'sch'
SCHEDULER_GAMMA: Final[str] = 'gamma'
FACTOR: Final[str] = 'factor'
START_FACTOR: Final[str] = 'start_factor'
END_FACTOR: Final[str] = 'end_factor'
T_MAX: Final[str] = 't_max'
T_MULT: Final[str] = 't_mult'
ETA_MIN: Final[str] = 'eta_min'
MILESTONES: Final[str] = 'milestones'
POWER: Final[str] = 'power'
SCHEDULER_STEP_SIZE: Final[str] = 'step_size'
SCHEDULER_STEP_TYPE: Final[str] = 'step_type'
SCHEDULER_STEP_TYPE_AVAILABLE: Final[List[str]] = ['epoch', 'step']

#============================= logs =============================#
LOGS_FOLDER: Final[str] = os.path.join(DATA_PATH, 'logs')
LOGGERS_CONFIG_FILE: Final[str] = os.path.join(DATA_PATH, 'loggerConfig.json')
APP_LOGGER: Logger | None = None

APP_LOGGER_NAME: Final[str] = 'appInfo'
CONSOLE_LOGGER_NAME: Final[str] = 'console'