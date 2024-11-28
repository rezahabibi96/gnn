import time
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from helpers import Config
from helpers import Log
from helpers import Tb
from dataloader import TrafficDataset, split_data, get_ctxs, get_joint
from models import ST_GAT, model_train, model_eval, save_checkpoint, load_checkpoint


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Config.PARAMS.CUDA['DEVICE'] = device 
    Log.info("using {}".format(Config.PARAMS.CUDA['DEVICE']))
    
    if Config.PARAMS.ACTIVE['LEARNING'] == 'SL':
        Log.info("running SL on {}".format(Config.PARAMS.ACTIVE['DATA']))
        dataset = TrafficDataset()
        train, val, test = split_data(dataset, (34, 5, 5))

        train_dataloader = DataLoader(train, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
        val_dataloader = DataLoader(val, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
        test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

        model = ST_GAT(in_channels=Config.PARAMS.HYPER['N_HIST'], out_channels=Config.PARAMS.HYPER['N_PRED'],
                    n_nodes=Config.PARAMS.DATA[Config.PARAMS.ACTIVE['DATA']]['N_NODES'], 
                    dropout=Config.PARAMS.HYPER['DROPOUT'])
        optim_fn = optim.Adam(model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                            weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])    
        loss_fn = torch.nn.MSELoss
        time_strf = time.strftime("%Y-%m-%d %H:%M:%S")
        
        Tb.begin(time_strf, '.standard_model')

        model_train(model, train_dataloader, val_dataloader, optim_fn, loss_fn)
        model_eval(model, test_dataloader)

    elif Config.PARAMS.ACTIVE['LEARNING'] == 'CL':
        Log.info("running CL on {}".format(Config.PARAMS.ACTIVE['DATA']))
        dataset = TrafficDataset()

        # get the contexts object (consist of training and testing dataset per context)
        ctxs = get_ctxs(dataset)

        # get the join object (consist of whole combined training dataset from contexts)
        joint = get_joint(dataset)
        joint_dataloader = DataLoader(joint, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

        # train and test the joint model (upper bound) on whole contexts dataset
        joint_model = ST_GAT(in_channels=Config.PARAMS.HYPER['N_HIST'], out_channels=Config.PARAMS.HYPER['N_PRED'],
                             n_nodes=Config.PARAMS.DATA[Config.PARAMS.ACTIVE['DATA']]['N_NODES'], 
                             dropout=Config.PARAMS.HYPER['DROPOUT'])
        joint_optim_fn = optim.Adam(joint_model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                                    weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])    
        loss_fn = torch.nn.MSELoss  
        time_strf = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # train joint model (upper bound) on whole contexts dataset
        Tb.begin(time_strf, '.joint_model')
        joint_model, joint_loss = model_train(joint_model, joint_dataloader, joint_dataloader, 
                                              joint_optim_fn, loss_fn)
        save_checkpoint(joint_model, joint_optim_fn, joint_loss, time_strf, 
                        'joint_model', joint_dataloader.dataset.name)

        # test the joint model (upper bound) per context
        for ctx in ctxs:
            id = ctx['id']

            test = ctx['test']
            test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
            
            Tb.begin(time_strf, f'.joint_model_c{id}')
            model_eval(joint_model, test_dataloader)

        # train the naive model on per context and test it on per context
        ctx_model = ST_GAT(in_channels=Config.PARAMS.HYPER['N_HIST'], out_channels=Config.PARAMS.HYPER['N_PRED'],
                           n_nodes=Config.PARAMS.DATA[Config.PARAMS.ACTIVE['DATA']]['N_NODES'], 
                           dropout=Config.PARAMS.HYPER['DROPOUT'])
        ctx_optim_fn = optim.Adam(ctx_model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                                  weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])

        # train the naive model per context
        for i, ctx in enumerate(ctxs):
            id_a = ctx['id']

            train = ctx['train']
            train_dataloader = DataLoader(train, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

            Tb.begin(time_strf, f'.c{id_a}_model')
            ctx_model, ctx_loss = model_train(ctx_model, train_dataloader, train_dataloader, 
                                              ctx_optim_fn, loss_fn)
            save_checkpoint(ctx_model, ctx_optim_fn, ctx_loss, time_strf, 
                            f'c{id_a}_model', train_dataloader.dataset.name)
            
            # test the naive model per context
            for j, ctx in enumerate(ctxs):
                if j>i:
                    break

                id_b = ctx['id']

                test = ctx['test']
                test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

                Log.info(f'.c{id_a}_model_c{id_b}')

                Tb.begin(time_strf, f'.c{id_a}_model_c{id_b}')
                model_eval(ctx_model, test_dataloader)


if __name__ == "__main__":
    main()