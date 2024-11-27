import os
import time
import torch

from helpers import Config
from helpers import Log


def save_checkpoint(epoch, model, optim_fn, loss):
    """
    save model given checkpoint.

    :param epocj: epoch value.
    :param model: trained model.
    :param optim_fn: optim function.
    :param loss: loss value. 
    """

    time_strf = time.strftime("%Y-%m-%d %H:%M:%S")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim_fn.state_dict(),
        "loss": loss,
    }, os.path.join(Config.PARAMS.DIR['CHECKPOINTS'], f'ckpt_{time_strf}'))

def load_checkpoint(model, path):
    """
    load model from checkpoint.

    :param model: model structure.
    :param path: checkpoint path. 

    :return model: loaded model.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    return model