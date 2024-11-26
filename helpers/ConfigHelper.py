import os
import yaml
from pydantic import BaseModel
from yaml.loader import Loader
from jsonargparse import ArgumentParser

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
    IS_GAT: bool = False
    N_NODES: int = 228
    N_DAYS: int = 44
    N_INTERVALS: int = 288 # possible number of five minutes intervals per day
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
            ENVIRONTMENT=cls.__config_yaml__['env'],
            BATCH_SIZE=cls.__config_yaml__['model']['BATCH_SIZE'],
            N_EPOCHS=cls.__config_yaml__['model']['N_EPOCHS'],
            P_LR=cls.__config_yaml__['model']['P_LR'],
            P_DROPOUT=cls.__config_yaml__['model']['P_DROPOUT'],
            P_WEIGHT_DECAY=cls.__config_yaml__['model']['P_WEIGHT_DECAY'],
            N_HIST=cls.__config_yaml__['model']['N_HIST'],
            N_PRED=cls.__config_yaml__['model']['N_PRED'],
            IS_GAT=cls.__config_yaml__['model']['IS_GAT'],
            N_NODES=cls.__config_yaml__['model']['N_NODES'],
            N_DAYS=cls.__config_yaml__['model']['N_DAYS'],
            N_INTERVALS=cls.__config_yaml__['model']['N_INTERVALS'],
        )

        cls.PARAMS.N_SLOTS = cls.PARAMS.N_INTERVALS - (cls.PARAMS.N_HIST + cls.PARAMS.N_PRED) + 1

    @classmethod
    def parse_cli(cls):
        Log.info("parse config from cli")

        parser = ArgumentParser(description="Configuration CLI Parser (CLI overrides YAML defaults)", 
                                default_meta=True)
        
        parser.add_argument('--ENVIRONTMENT', '--env', type=str, default=cls.PARAMS.ENVIRONTMENT,
            help="environment type (e.g., development, production)")

        parser.add_argument('--model.BATCH_SIZE', '--batch_size', type=int, default=cls.PARAMS.BATCH_SIZE,
            help="batch size for training")

        parser.add_argument('--model.N_EPOCHS', '--epochs', type=int, default=cls.PARAMS.N_EPOCHS,
            help="total epochs for training")

        parser.add_argument('--model.P_LR', '--lr', type=float, default=cls.PARAMS.P_LR,
            help="learning rate")

        parser.add_argument('--model.P_DROPOUT', '--dropout', type=float, default=cls.PARAMS.P_DROPOUT,
            help="dropout rate")
        
        parser.add_argument('--model.P_WEIGHT_DECAY', '--weight_decay', type=float, default=cls.PARAMS.P_WEIGHT_DECAY,
            help="weight decay rate")
        
        parser.add_argument('--model.N_HIST', '--hist', type=int, default=cls.PARAMS.N_HIST,
            help="number of hist")
        
        parser.add_argument('--model.N_PRED', '--pred', type=int, default=cls.PARAMS.N_PRED,
            help="number of pred")
        
        parser.add_argument('--model.IS_GAT', '--is_gat', type=bool, default=cls.PARAMS.IS_GAT,
            help="enable Graph Attention Network mode")
        
        # parser.add_argument('--model.N_NODES', '--nodes', type=int, default=cls.PARAMS.N_NODES,
        #     help="number of nodes")
        
        # parser.add_argument('--model.N_DAYS', '--days', type=int, default=cls.PARAMS.N_DAYS,
        #     help="number of days")
        
        # parser.add_argument('--model.N_INTERVALS', '--intervals', type=int, default=cls.PARAMS.N_INTERVALS,
        #     help="number of intervals")
        
        # parser.add_argument('--model.N_SLOTS', '--slots', type=int, default=cls.PARAMS.N_SLOTS,
        #     help="number of slots")

        cli_args = parser.parse_args()
        for field, value in cli_args.model.items():
            if value is not None:
                setattr(cls.PARAMS, field, value)

    @classmethod
    def load(cls, use_cli=True):
        cls.parse_yaml()

        if use_cli:
            cls.parse_cli()
        
        Log.info(f"final configuration \n {cls.PARAMS}")


Config.load()