import torch

from models.model import ST_GAT


def load_checkpoint(path_checkpoint, config):
    """
    load model from checkpoint.

    :param path_checkpoint: path to checkpoint. 
    :param config: configuration to use.

    :return model: loaded model.
    """

    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'],
                   n_nodes=config['N_NODES'])
    
    checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    return model