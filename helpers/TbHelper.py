import os
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter

from helpers import Config
from helpers import Log


summary_writer = SummaryWriter(log_dir=os.path.join(Config.PARAMS.DIR['TENSORBOARD'], '/temp'))


class Tb:
    summary_writer = summary_writer

    @classmethod
    def begin(cls, time_strf, filename_suffix, log_dir=None):
        cls.summary_writer.close()
        
        log_dir = os.path.join(Config.PARAMS.DIR['TENSORBOARD'], time_strf) 
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        cls.summary_writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        
    @classmethod
    def change_log_dir(cls, log_dir):
        cls.summary_writer.log_dir = log_dir
    
    @classmethod
    def change_filename_suffix(cls, filename_suffix):
        cls.summary_writer.filename_suffix = filename_suffix
    
    @classmethod
    def add_scalar(cls, tag, scalar_value, global_step):
        cls.summary_writer.add_scalar(tag, scalar_value, global_step)
    
    @classmethod
    def flush(cls):
        cls.summary_writer.flush()
    
    @classmethod
    def close(cls):
        cls.summary_writer.close()