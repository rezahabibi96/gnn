from helpers import Config
from models import ST_GAT


def create_model():
    """
    func to create model based on active model

    :return model: created model.
    """
    if Config.PARAMS.ACTIVE['MODEL'] == 'ST_GAT':
        model = ST_GAT(in_channels=Config.PARAMS.HYPER['N_HIST'], out_channels=Config.PARAMS.HYPER['N_PRED'],
                       n_nodes=Config.PARAMS.DATA[Config.PARAMS.ACTIVE['DATA']]['N_NODES'], 
                       dropout=Config.PARAMS.HYPER['DROPOUT'])
        
        return model