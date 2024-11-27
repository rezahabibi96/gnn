import os
import time
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from helpers import Config
from helpers import Log
from utils.math import *


# init a tensorboard writer
writer = SummaryWriter(log_dir=Config.PARAMS.DIR['TENSORBOARD'])


def train(model, dataloader, optim_fn, loss_fn, epoch):
    """
    train function.

    :param model: given model.
    :param dataloader: current dataloader.
    :param optim_fn: optim function.
    :param loss_fn: loss function.
    :param epoch: current epoch.

    :return loss: current loss.
    """
    model.train()

    with logging_redirect_tqdm(loggers=[Log.log]):
        for _, batch in enumerate(_ := tqdm(desc=Log.info(f'(train) EPOCH: {epoch}'), iterable=dataloader)):
            batch = batch.to(Config.PARAMS.CUDA['DEVICE'])
            optim_fn.zero_grad()

            y_pred = torch.squeeze(model(batch))
            loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())

            loss.backward()
            optim_fn.step()

            writer.add_scalar("LOSS/train", loss, epoch)
        
        return loss


@torch.no_grad()
def eval(model, dataloader):
    """
    eval function.

    :param model: given model.
    :param dataloader: given dataloader.

    :return: eval metrics (rmse, mae, mape) & data tensors (y_preds, y_truths).
    """
    model.eval()
    model.to(Config.PARAMS.CUDA['DEVICE'])

    rmse = 0
    mae = 0
    mape = 0
    n = 0

    # eval model on whole data
    for index, batch in enumerate(dataloader):
        batch = batch.to(Config.PARAMS.CUDA['DEVICE'])

        if batch.x.shape[0] == 1: # skip if batch contains only single data point
            pass
        else:
            with torch.no_grad():
                y_pred = model(batch) # y_pred value of curr batch
            
            y_truth = batch.y.view(y_pred.shape) # to reshape y_truth to match y_pred

            if index == 0:
                # init y_preds tensor to store all preds across batches
                y_preds = torch.zeros(len(dataloader), y_pred.shape[0], y_pred.shape[1])
                # init y_truths tensor to store all truths across batches
                y_truths = torch.zeros(len(dataloader), y_pred.shape[0], y_pred.shape[1])

            # denormalize y_truth and y_pred of curr batch
            y_truth = denorm_z(y_truth, dataloader.dataset.mean, dataloader.dataset.std_dev)
            y_pred = denorm_z(y_pred, dataloader.dataset.mean, dataloader.dataset.std_dev)

            # store truth and pred of curr batch
            y_preds[index, :y_pred.shape[0], :] = y_pred
            y_truths[index, :y_pred.shape[0], :] = y_truth

            # calc metrics
            rmse += calc_rmse(y_truth, y_pred)
            mae += calc_mae(y_truth, y_pred)
            mape += calc_mape(y_truth, y_pred)
            n += 1

    # avgs metrics
    rmse, mae, mape = rmse/n, mae/n, mape/n

    return rmse, mae, mape, y_preds, y_truths


def model_train(model, train_dataloader, val_dataloader, optim_fn, loss_fn):
    """
    train the given model.

    :param model: given model.
    :param train_dataloader: data loader of training dataset.
    :param val_dataloader: data loader of val dataset.
    :param optim_fn: optim function.
    :param loss_fn: loss function.

    :return model: trained model.
    """
    model.to(Config.PARAMS.CUDA['DEVICE'])

    # train model for each epoch
    for epoch in range(Config.PARAMS.HYPER['TOTAL_EPOCHS']):
        loss = train(model, train_dataloader, optim_fn, loss_fn, epoch)
        Log.info(f'(train) LOSS: {loss:.4f}')

        if epoch % 10 == 0:
            train_rmse, train_mae, train_mape, _, _ = eval(model, train_dataloader)
            Log.info(f'(train) RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}')

            val_rmse, val_mae, val_mape, _, _ = eval(model, val_dataloader)
            Log.info(f'(val) RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, MAPE: {val_mape:.4f}')

            writer.add_scalar(f"RMSE/train", train_rmse, epoch)
            writer.add_scalar(f"MAE/train", train_mae, epoch)
            writer.add_scalar(f"MAPE/train", train_mape, epoch)
            
            writer.add_scalar(f"RMSE/val", val_rmse, epoch)
            writer.add_scalar(f"MAE/val", val_mae, epoch)
            writer.add_scalar(f"MAPE/val", val_mape, epoch)

    writer.flush()
    time_str = time.strftime("%m-%d-%H%M%S")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim_fn.state_dict(),
        "loss": loss,
    }, os.path.join(Config.PARAMS.DIR['CHECKPOINTS'], f'model_{time_str}'))

    return model


def model_eval(model, test_dataloader):
    """
    test the given model.

    :param model: given model.
    :param test_dataloader: data loader of test dataset.
    """
    pass