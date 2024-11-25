import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
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
    SetUp TrafficDataset for GNN, it extends InMemoryDataset.
    """
    def __init__(self, W, config, root, transform=None, pre_transform=None, pre_filter=None):
        self.W = W
        self.config = config
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, 'PeMSD7_V_228.csv')]
    
    @property
    def processed_file_names(self):
        return ['./data.pt']
    
    def download(self):
        copyfile('./data/PeMSD7_V_228.csv', os.path.join(self.raw_dir, 'PeMSD7_V_228.csv'))
    
    def process(self):
        """
        To process the raw dataset into .pt dataset for the later use.
        Please note that any property (self.fields) here would not exist, 
        If it loads straight from the .pt dataset.
        """

        # load and process dataset
        data = pd.read_csv(self.raw_file_names[0], header=False).values
        mean = np.mean(data)
        std = np.std(data)

        n_nodes = data.shape[-1]

        # create tensor for possible number of edges (n_nodes x n_nodes) 
        edge_index = torch.zeros((2, n_nodes**2), dtype=torch.long)
        edge_attr = torch.zeros((n_nodes**2, 1))

        n_edges = 0 # to store the actual number of edges
        for i in range(n_nodes):
            for j in range(n_nodes):

                # if weight/distance of node i and node j != 0
                if self.W[i, j] != 0.:

                    # create edge index between node i and node j
                    edge_index[0, n_edges] = i
                    edge_index[1, n_edges] = j

                    # fill edge attr with its weight/distance
                    edge_attr[n_edges, 1] = self.W[i, j]
                    n_edges += 1
        
        # keep only the actual number of edges (n_edges)
        edge_index = edge_index.resize_(2, n_edges)
        edge_attr = edge_attr.resize_(n_edges, 1)

        # to store sequence/collection of graph
        seqs = []
        window = self.config['N_HIST'] + self.config['N_PRED']
        
        # possible number of windows per day
        N_SLOTS = self.config['N_INTERVALS'] - window + 1

        # construct graph for each window
        for i in range(self.config['N_DAYS']):
            for j in range(self.config['N_SLOTS']):

                g = Data()
                g.num_nodes = n_nodes

                g.edge_index = edge_index
                g.edge_attr = edge_attr

                start = i * self.config['N_INTERVALS'] + j
                end = start + window

                # switch from [F, N] (21, 228) -> [N, F] (228, 21)
                data_window = np.swapaxes(data[start:end, :], 0, 1) 

                # X feature vector for each node
                g.x = torch.FloatTensor(data_window[:, 0:self.config['N_HIST']])
                # Y ground truth for each node
                g.y = torch.FloatTensor(data_window[:, self.config['N_HIST']::])

                seqs += [g]
        
        # construct the actual dataset from sequence/collection of graph 
        data, slices = self.collate(seqs)
        torch.save((data, slices, n_nodes, mean, std), self.processed_paths[0])


def split_data(data, n_slots, ratio):
    """
    given data, split it into subsets of train, val, and test.

    :param data: data to split.
    :param n_slots: possible number of sliding windows in a day.
    :param ratio: (train, val, test) ratio.

    :return train, val, test: splitted data.
    """
    r_train, r_val, _ = ratio

    n_train = n_slots * r_train
    n_val = n_slots * r_val

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test