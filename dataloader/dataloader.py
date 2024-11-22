import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from shutil import copyfile

from utils import math


def dist_to_weight(D, sigma2=0.1, epsilon=0.5, gat=False):
    """
    given D, distance matrix between all nodes, convert it into weight matrix W.

    :param D: distance matrix.
    :param sigma2: user configurable parameter to adjust the sparsity.
    :param epsilon: user configurable parameter to adjust the sparsity.
    :param gat: if true, use 0/1 weights with self-loop.

    :return W: weight matrix.
    """
    n = D.shape[0]
    D = D/1000

    D2 = D*D
    W_mask = np.ones([n, n]) - np.identity(n)

    # refer to eq 10 from paper
    W_temp = np.exp(-D2 / sigma2)
    W = W_temp * (W_temp >= epsilon)*W_mask

    # if using gat, round to 0/1 and include self-loop.
    if gat:
        W[W>0] = 1
        W += np.identity(n)
    
    return W


class TrafficDataset(InMemoryDataset):
    """
    TrafficDataset for GNN, it extends InMemoryDataset
    """
    def __init__(self, W, config, root, transform=None, pre_transform=None, pre_filter=None):
        self.W = W
        self.config = config
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])