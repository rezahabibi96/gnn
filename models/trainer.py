import os
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from models.model_st_gat import ST_GAT
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
    for index, batch in enumerate(dataloader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1: # skip if batch contains only single data point
            pass
        else:
            with torch.no_grad():
                y_pred = model(batch, device) # y_pred value of curr batch
            
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

    print(f'{type}, RMSE: {rmse}, MAE: {mae}')

    return rmse, mae, mape, y_pred, y_truth


def model_train(train_dataloader, val_dataloader, config, device):
    """
    train the given model.

    :param train_dataloader: data loader of training dataset.
    :param val_dataloader: data loader of val dataset.
    :param config: configuration to use.
    :param device: device to use.

    :return model: trained model.
    """
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'],
                   n_nodes=config['N_NODES'], dropout=config['P_DROPOUT'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['P_LR'], 
                           weight_decay=config['P_WEIGHT_DECAY'])
    
    loss_fn = torch.nn.MSELoss

    model.to(device)

    # train model for each epoch
    for epoch in range(config['N_EPOCHS']):
        loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        print(f'loss: {loss:.3f}')

        if epoch % 5 == 0:
            train_rmse, train_mae, train_mape, _, _ = eval(model, device, train_dataloader, 'train')
            val_rmse, val_mae, val_mape, _, _ = eval(model, device, val_dataloader, 'val')

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
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, os.path.join(config['DIR_CHECKPOINT'], f'model_{time_str}'))

    return model


def model_eval(model, test_dataloader, device, config):
    """
    test the given model.

    :param model: given model.
    :param test_dataloader: data loader of test dataset.
    :param config: configuration to use.
    :param device: device to use.
    """
    pass