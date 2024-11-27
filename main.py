import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from helpers import Config
from helpers import Log
from dataloader import TrafficDataset, split_data
from models import ST_GAT, model_train


def main():
    Log.info("running {}".format(Config.PARAMS.ACTIVE_DATA))
    dataset = TrafficDataset()
    train, val, test = split_data(dataset, (34, 5, 5))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Config.PARAMS.CUDA['DEVICE'] = device 
    Log.info("using {}".format(Config.PARAMS.CUDA['DEVICE']))

    train_dataloader = DataLoader(train, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

    model = ST_GAT(in_channels=Config.PARAMS.HYPER['N_HIST'], out_channels=Config.PARAMS.HYPER['N_PRED'],
                   n_nodes=Config.PARAMS.DATA[Config.PARAMS.ACTIVE_DATA]['N_NODES'], 
                   dropout=Config.PARAMS.HYPER['DROPOUT'])
    optim_fn = optim.Adam(model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                           weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])    
    loss_fn = torch.nn.MSELoss

    model_train(model, train_dataloader, val_dataloader, optim_fn, loss_fn)


if __name__ == "__main__":
    main()