import os
import yaml
from pydantic import BaseModel
from yaml.loader import Loader
from jsonargparse import ArgumentParser

from helpers.LogHelper import Log


class ConfigModel(BaseModel):
    # ENV
    ENV: str = None
    
    # MODEL
    BATCH_SIZE: int = None
    TOTAL_EPOCHS: int = None
    LEARNING_RATE: float = None
    DROPOUT: float = None
    WEIGHT_DECAY: float = None
    IS_GAT: bool = None
    N_HIST: int = None
    N_PRED: int = None

    # DIR
    CHECKPOINT: str = None
    LOG: str = None

    # DATA -> PEMSD7
    N_NODES: int = None
    N_DAYS: int = None
    N_INTERVALS: int = None # possible number of five minutes intervals per day
    N_SLOTS: int = None # possible number of windows per day (derived value)


class Config:
    __dir_name__ = os.path.dirname(__file__)
    __file_path__ = '../configs/config.yaml'
    __file_config__ = os.path.abspath(os.path.join(__dir_name__, __file_path__))
    __config_yaml__ = None

    PARAMS = ConfigModel

    @classmethod
    def parse_yaml(cls):
        config = open(cls.__file_config__, "r")
        cls.__config_yaml__ = yaml.load(config, Loader=Loader)
        
        Log.info("load config from file {}".format(cls.__file_config__))

        cls.PARAMS = ConfigModel(
            ENV=cls.__config_yaml__['ENV'],
            
            BATCH_SIZE=cls.__config_yaml__['MODEL']['BATCH_SIZE'],
            TOTAL_EPOCHS=cls.__config_yaml__['MODEL']['TOTAL_EPOCHS'],
            LEARNING_RATE=cls.__config_yaml__['MODEL']['LEARNING_RATE'],
            DROPOUT=cls.__config_yaml__['MODEL']['DROPOUT'],
            WEIGHT_DECAY=cls.__config_yaml__['MODEL']['WEIGHT_DECAY'],
            IS_GAT=cls.__config_yaml__['MODEL']['IS_GAT'],
            N_HIST=cls.__config_yaml__['MODEL']['N_HIST'],
            N_PRED=cls.__config_yaml__['MODEL']['N_PRED'],

            CHECKPOINT=cls.__config_yaml__['DIR']['CHECKPOINT'],
            LOG=cls.__config_yaml__['DIR']['LOG'],

            N_NODES=cls.__config_yaml__['DATA']['PEMSD7']['N_NODES'],
            N_DAYS=cls.__config_yaml__['DATA']['PEMSD7']['N_DAYS'],
            N_INTERVALS=cls.__config_yaml__['DATA']['PEMSD7']['N_INTERVALS'],
        )

        cls.PARAMS.N_SLOTS=cls.PARAMS.N_INTERVALS - (cls.PARAMS.N_HIST + cls.PARAMS.N_PRED) + 1

    @classmethod
    def parse_cli(cls):
        Log.info("parse config from cli")

        parser = ArgumentParser(description="Configuration CLI Parser (CLI overrides YAML defaults)", 
                                default_meta=True)
        
        parser.add_argument('--ENV', '-e', type=str, default=cls.PARAMS.ENV,
            help="environment type (e.g., development, production)")
        
        model_group = parser.add_argument_group("MODEL", "model-related parameters")
        model_group.add_argument('--BATCH_SIZE', type=int, default=cls.PARAMS.BATCH_SIZE, 
                                 help="batch size")
        model_group.add_argument('--TOTAL_EPOCHS', type=int, default=cls.PARAMS.TOTAL_EPOCHS, 
                                 help="total epochs")
        model_group.add_argument('--LEARNING_RATE', type=float, default=cls.PARAMS.LEARNING_RATE, 
                                 help="learning rate")
        model_group.add_argument('--DROPOUT', type=float, default=cls.PARAMS.DROPOUT, 
                                 help="dropout rate")
        model_group.add_argument('--WEIGHT_DECAY', type=float, default=cls.PARAMS.WEIGHT_DECAY, 
                                 help="weight decay")
        model_group.add_argument('--IS_GAT', type=bool, default=cls.PARAMS.IS_GAT, 
                                 help="gat mode")
        model_group.add_argument('--N_HIST', type=int, default=cls.PARAMS.N_HIST, 
                                 help="number of history intervals")
        model_group.add_argument('--N_PRED', type=int, default=cls.PARAMS.N_PRED, 
                                 help="number of prediction intervals")
        
        dir_group = parser.add_argument_group("DIR", "directory-related parameters")
        dir_group.add_argument('--CHECKPOINT', type=str, default=cls.PARAMS.CHECKPOINT, 
                               help="checkpoint directory")
        dir_group.add_argument('--LOG', type=str, default=cls.PARAMS.LOG, 
                               help="log directory")
        
        data_group = parser.add_argument_group("DATA", "data-related parameters")
        data_group.add_argument('--N_NODES', type=int, default=cls.PARAMS.N_NODES,
                                help="number of nodes")
        data_group.add_argument('--N_DAYS', type=int, default=cls.PARAMS.N_DAYS,
                                help="number of days")
        
        data_group.add_argument('--N_INTERVALS', type=int, default=cls.PARAMS.N_INTERVALS,
                                help="number of intervals")
        
        data_group.add_argument('--N_SLOTS', type=int, default=cls.PARAMS.N_SLOTS,
                                help="number of slots")

        cli_args = parser.parse_args()
        for field, value in vars(cli_args).items():
            if value is not None:
                setattr(cls.PARAMS, field, value)

    @classmethod
    def load(cls, use_cli=True):
        cls.parse_yaml()

        if use_cli:
            cls.parse_cli()
        
        Log.info(f"final configuration \n {cls.PARAMS}")


Config.load()