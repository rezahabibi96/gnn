import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from models.model import ST_GAT
from utils.math import *


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

@torch.no_grad
def eval(model, device, dataloader, type=''):
    """
    eval function.

    :param model: model.
    :param device: device.
    :param dataloader: dataloader.
    :param type: type (train/val/test).

    :return: eval metrics (rmse, mae, mape) & data tensors (y_pred, y_truth).
    """
    model.eval()
    model.to(device)

    rmse = 0
    mae = 0
    mape = 0
    n = 0

    # eval model on whole data
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1: # skip if batch contains only single data point
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device)
            
            truth = batch.y.view(pred.shape) # to reshape truth to match pred

            if i == 0:
                # init y_pred tensor to store all preds across batches
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                # init y_truth tensor to store all truths across batches
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])

            # denormalize truth and pred of curr batch
            truth = denorm_z(truth, dataloader.dataset.mean, dataloader.dataset.std_dev)
            pred = denorm_z(pred, dataloader.dataset.mean, dataloader.dataset.std_dev)

            # store truth and pred of curr batch
            y_pred[i, :pred.shape[0], :] = pred
            y_truth[i, :pred.shape[0], :] = truth

            # calc metrics
            rmse += calc_rmse(truth, pred)
            mae += calc_mae(truth, pred)
            mape += calc_mape(truth, pred)
            n += 1

    # avgs metrics
    rmse, mae, mape = rmse/n, mae/n, mape/n

    print(f'{type}, RMSE: {rmse}, MAE: {mae}')

    return rmse, mae, mape, y_pred, y_truth