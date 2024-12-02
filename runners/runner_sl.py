import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from helpers import Config, Log, Tb
from factories import create_dataset, create_model
from dataloaders import split_data
from trainers import model_train, model_eval


def run_sl(time_strf):
    Log.info("running SL on {}".format(Config.PARAMS.ACTIVE['DATA']))
    
    dataset = create_dataset()
    train, val, test = split_data(dataset, (34, 5, 5))

    train_dataloader = DataLoader(train, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

    model = create_model()
    optim_fn = optim.Adam(model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                          weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])    
    loss_fn = torch.nn.MSELoss

    Tb.begin(time_strf, '.standard_model')

    model_train(model, train_dataloader, val_dataloader, optim_fn, loss_fn)
    model_eval(model, test_dataloader)