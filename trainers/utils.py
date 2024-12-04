import random
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn


def set_seed(seed=0):
    """
    func to set seed for reproducibility
    """
    
    max_seed = (1 << 32) - 1
    
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    
    torch.backends.cudnn.benchmark = False  # if benchmark = True deterministic will be False
    torch.backends.cudnn.deterministic = True