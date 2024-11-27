import os
import yaml
from pydantic import BaseModel
from yaml.loader import Loader
from jsonargparse import ArgumentParser

from helpers.LogHelper import Log


class ConfigModel(BaseModel):
    ACTIVE_ENVIRONTMENT: str = None
    ACTIVE_DATA: str = None
    CUDA: dict = {}
    DATA: dict = {}
    DIR: dict = {}
    HYPER: dict = {}
    ST_GAT: dict = {}
    

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
            ACTIVE_ENVIRONTMENT=cls.__config_yaml__['ACTIVE_ENVIRONTMENT'],
            ACTIVE_DATA=cls.__config_yaml__['ACTIVE_DATA'],
            CUDA=cls.__config_yaml__['CUDA'],
            DATA=cls.__config_yaml__['DATA'],
            DIR=cls.__config_yaml__['DIR'],
            HYPER=cls.__config_yaml__['MODEL']['HYPER'],
            ST_GAT=cls.__config_yaml__['MODEL']['ST_GAT'],
        )

        cls.PARAMS.DATA[cls.PARAMS.ACTIVE_DATA]['N_SLOTS'] = ( cls.PARAMS.DATA[cls.PARAMS.ACTIVE_DATA]['N_INTERVALS'] - 
                                                              (cls.PARAMS.HYPER['N_HIST'] + cls.PARAMS.HYPER['N_PRED']) 
                                                              + 1 )

    @classmethod
    def parse_cli(cls):
        Log.info("parse config from cli")

        parser = ArgumentParser(description="Configuration CLI Parser (CLI overrides YAML defaults)", 
                                default_meta=True)
        
        active_group = parser.add_argument_group("ACTIVE", "active-related") 
        
        active_group.add_argument('--ACTIVE_ENVIRONTMENT', '-ae', type=str, default=cls.PARAMS.ACTIVE_ENVIRONTMENT,
                               help="select environtment (e.g., development, production)")
        active_group.add_argument('--ACTIVE_DATA', '-ad', type=str, default=cls.PARAMS.ACTIVE_DATA,
                               help="select data (e.g., PEMSD7, PEMSD8)")

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