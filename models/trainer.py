import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from models.model import ST_GAT
from utils import *


# init a tensorboard writer
writer = SummaryWriter()

def train(model, device, dataloader, optimizier, loss_fn, epoch):
    """
    train function.

    :param model: model.
    :param device: device.
    :param dataloader: dataloader.
    :param optimizer: optimizer.
    :param loss_fn: loss function.
    :param epoch: current epoch.

    :return loss: current loss.
    """
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch}")):
        batch = batch.to(device)
        optimizier.zero_grad()

        y_pred = torch.squeeze(model(batch, device))
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())

        loss.backward()
        optimizier.step()

        writer.add_scalar("loss/train", loss, epoch)
    
    return loss