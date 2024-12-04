import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from helpers import Config, Log, Tb
from factories import create_dataset, create_model
from dataloaders import get_ctxs, get_joint
from trainers import model_train, model_eval, save_checkpoint


def run_cl(time_strf):
    Log.info("running CL on {}".format(Config.PARAMS.ACTIVE['DATA']))
    
    dataset = create_dataset()

    # get the contexts object (consist of training and testing dataset per context)
    ctxs = get_ctxs(dataset)

    # get the join object (consist of whole combined training dataset from contexts)
    joint = get_joint(dataset)
    joint_dataloader = DataLoader(joint, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

    # init the joint model (upper bound)
    joint_model = create_model()
    joint_optim_fn = optim.Adam(joint_model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                                weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])    
    loss_fn = torch.nn.MSELoss  
    
    # train joint model (upper bound) on whole contexts dataset
    Log.info('cl_joint_model')
    Tb.begin(time_strf, '.cl_joint_model')
    joint_model, joint_loss = model_train(joint_model, joint_dataloader, joint_dataloader, 
                                          joint_optim_fn, loss_fn)
    save_checkpoint(joint_model, joint_optim_fn, joint_loss, time_strf, 
                    'cl_joint_model', joint_dataloader.dataset.name)

    # test the joint model (upper bound) per context
    for ctx in ctxs:
        id = ctx['id']

        test = ctx['test']
        test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)
        
        Log.info(f'cl_joint_model_on_c{id}')
        Tb.begin(time_strf, f'.cl_joint_model_on_c{id}')
        model_eval(joint_model, test_dataloader)

    # init the naive model
    ctx_model = create_model()
    ctx_optim_fn = optim.Adam(ctx_model.parameters(), lr=Config.PARAMS.HYPER['LEARNING_RATE'], 
                              weight_decay=Config.PARAMS.HYPER['WEIGHT_DECAY'])


    # train the naive model per context
    for i, ctx in enumerate(ctxs):
        id_a = ctx['id']

        train = ctx['train']
        train_dataloader = DataLoader(train, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

        Log.info(f'cl_c{id_a}_model')
        Tb.begin(time_strf, f'.cl_c{id_a}_model')
        ctx_model, ctx_loss = model_train(ctx_model, train_dataloader, train_dataloader, 
                                          ctx_optim_fn, loss_fn)
        save_checkpoint(ctx_model, ctx_optim_fn, ctx_loss, time_strf, 
                        f'cl_c{id_a}_model', train_dataloader.dataset.name)
        
        # test the naive model on its trained context and its previous contexts
        for j, ctx in enumerate(ctxs):
            if j>i:
                break

            id_b = ctx['id']

            test = ctx['test']
            test_dataloader = DataLoader(test, batch_size=Config.PARAMS.HYPER['BATCH_SIZE'], shuffle=True)

            Log.info(f'cl_c{id_a}_model_on_c{id_b}')
            Tb.begin(time_strf, f'.c{id_a}_model_on_c{id_b}')
            model_eval(ctx_model, test_dataloader)