import os
import torch
from pathlib import Path

from helpers import Config


def save_checkpoint(model, optim_fn, loss, time_strf, file_name, data_name):
    """
    save model given checkpoint.

    :param model: trained model.
    :param optim_fn: optim function.
    :param loss: loss value.
    :param time_strf: time stringf.
    :param file_name: file name.
    :param data_name: data name.
    """
    model_path = os.path.join(Config.PARAMS.DIR['CHECKPOINTS'], time_strf) 
    Path(model_path).mkdir(parents=True, exist_ok=True)

    detail = {
        "model": model.__class__.__name__,
        "time_strf": time_strf,
        "data": data_name,
        "loss": loss,
        "hyper": Config.PARAMS.HYPER
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim_fn.state_dict(),
        "detail": detail
    }, os.path.join(model_path, f'{file_name}.pt'))


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