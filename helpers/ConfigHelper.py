import os
import yaml
from pydantic import BaseModel
from yaml.loader import Loader

from helpers.LogHelper import Log


class ConfigModel(BaseModel):
    ENVIRONTMENT: str = ''
    BATCH_SIZE: int = 50
    N_EPOCHS: int = 200
    P_LR: float = 3e-4
    P_DROPOUT: float = 0.2
    P_WEIGHT_DECAY: float = 5e-5
    N_HIST: int = 9
    N_PRED: int = 12
    N_DAYS: int = 44
    N_INTERVALS: int = 288 # possible number of five minutes intervals per day
    N_SLOTS: int = 0 # possible number of windows per day
    N_NODES: int = 228
    IS_GAT: bool = True


class Config:
    __dir_name__ = os.path.dirname(__file__)
    __file_path__ = '../configs/config.yaml'
    __file_config__ = os.path.abspath(os.path.join(__dir_name__, __file_path__))

    __config_yaml__ = None
    PARAMS = ConfigModel

    @classmethod
    def load(cls):
        config = open(cls.__file_config__, "r")
        cls.__config_yaml__ = yaml.load(config, Loader=Loader)
        
        Log.info("load config/config.yaml")

        cls.PARAMS = ConfigModel(
            ENVIRONTMENT=cls.__config_yaml__['env'],
            BATCH_SIZE=cls.__config_yaml__['model']['BATCH_SIZE'],
            N_EPOCHS=cls.__config_yaml__['model']['N_EPOCHS'],
            P_LR=cls.__config_yaml__['model']['P_LR'],
            P_DROPOUT=cls.__config_yaml__['model']['P_DROPOUT'],
            P_WEIGHT_DECAY=cls.__config_yaml__['model']['P_WEIGHT_DECAY'],
            N_HIST=cls.__config_yaml__['model']['N_HIST'],
            N_PRED=cls.__config_yaml__['model']['N_PRED'],
            N_INTERVALS=cls.__config_yaml__['model']['N_INTERVALS'],
            N_NODES=cls.__config_yaml__['model']['N_NODES'],
            IS_GAT=cls.__config_yaml__['model']['IS_GAT']   
        )

        cls.PARAMS.N_SLOTS = cls.PARAMS.N_INTERVALS - (cls.PARAMS.N_HIST + cls.PARAMS.N_PRED) + 1

        Log.info("environment config '{}' has loaded".format(cls.__config_yaml__['env']))


Config.load()
