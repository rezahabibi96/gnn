from torch.utils.tensorboard.writer import SummaryWriter

from helpers import Config
from helpers import Log


summary_writer = SummaryWriter(log_dir=Config.PARAMS.DIR['TENSORBOARD'])


class Tb:
    summary_writer = summary_writer

    @classmethod
    def begin(cls, log_dir=None):
        if log_dir:
            cls.summary_writer = SummaryWriter(log_dir=log_dir)
        else:
            cls.summary_writer = SummaryWriter(log_dir=Config.PARAMS.DIR['TENSORBOARD'])

    @classmethod
    def change_log_dir(cls, log_dir):
        cls.summary_writer.log_dir = log_dir
    
    @classmethod
    def add_scalar(cls, tag, scalar_value, global_step):
        cls.summary_writer.add_scalar(tag, scalar_value, global_step)
    
    @classmethod
    def flush(cls):
        cls.summary_writer.flush()
    
    @classmethod
    def close(cls):
        cls.summary_writer.close()