import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data

from helpers import Config
from utils import *


def from_dist_to_weight(D, sigma2=0.1, epsilon=0.5, gat_mode=False):
    """
    given D, distance matrix between all nodes, convert it into weight matrix W.

    :param D: distance matrix.
    :param sigma2: user configurable parameter to adjust the sparsity.
    :param epsilon: user configurable parameter to adjust the sparsity.
    :param gat: if true, use 0/1 weights with self-loop.

    :return W: weight matrix.
    """
    n = D.shape[0]
    D = D/1000 # for numerical stability purpose

    D2 = D*D
    W_mask = np.ones([n, n]) - np.identity(n)

    # refer to eq 10 from paper
    W_temp = np.exp(-D2 / sigma2)
    W = W_temp * (W_temp >= epsilon)*W_mask

    # if using gat, round to 0/1 and include self-loop.
    if gat_mode:
        W[W>0] = 1
        W += np.identity(n)
    
    return W


class TrafficDataset(InMemoryDataset):
    """
    SetUp TrafficDataset for GNN, it extends InMemoryDataset.
    """
    def __init__(self, root='', transform=None, pre_transform=None, pre_filter=None):
        self.supported = ['PEMSD7',]
        self.source = 'PEMSD7'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # PyG<2.4
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.mean, self.std, self.name, self.n_nodes = torch.load(self.processed_paths[1])
    
    @property
    def raw_dir(self):
        return Config.PARAMS.DATA[self.source]['RAW_DIR']

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, Config.PARAMS.DATA[self.source]['W']),
                os.path.join(self.raw_dir, Config.PARAMS.DATA[self.source]['V'])]
    
    @property
    def processed_dir(self):
        return Config.PARAMS.DATA[self.source]['PROCESSED_DIR']

    @property
    def processed_file_names(self):
        return ['data.pt', 'metadata.pt']
    
    def download(self):
        # from shutil import copyfile
        # copyfile('', '')
        pass
    
    def process(self):
        """
        To process the raw dataset into .pt dataset for the later use.
        Please note that any property (self.fields) here would not exist, 
        If it loads straight from the .pt dataset.
        """

        # calc W given D
        D = pd.read_csv(self.raw_file_names[0], header=None).values
        W = from_dist_to_weight(D, gat_mode=Config.PARAMS.HYPER['GAT_MODE'])

        # load and process dataset
        data = pd.read_csv(self.raw_file_names[1], header=None).values
        mean = np.mean(data)
        std = np.std(data)
        data = norm_z(data, mean, std)
        
        n_nodes = data.shape[-1]

        # create tensor for possible number of edges (n_nodes x n_nodes) 
        edge_index = torch.zeros((2, n_nodes**2), dtype=torch.long)
        edge_attr = torch.zeros((n_nodes**2, 1))

        n_edges = 0 # to store the actual number of edges
        for i in range(n_nodes):
            for j in range(n_nodes):

                # if weight/distance of node i and node j != 0
                if W[i, j] != 0.:

                    # create edge index between node i and node j
                    edge_index[0, n_edges] = i
                    edge_index[1, n_edges] = j

                    # fill edge attr with its weight/distance
                    edge_attr[n_edges] = W[i, j]
                    n_edges += 1
        
        # keep only the actual number of edges (n_edges)
        edge_index = edge_index.resize_(2, n_edges)
        edge_attr = edge_attr.resize_(n_edges, 1)

        # to store sequence/collection of graph
        seqs = []
        window = Config.PARAMS.HYPER['N_HIST'] + Config.PARAMS.HYPER['N_PRED']

        # construct graph for each window
        for i in range(Config.PARAMS.DATA[self.source]['N_DAYS']):
            for j in range(Config.PARAMS.DATA[self.source]['N_SLOTS']):

                g = Data()
                g.num_nodes = n_nodes

                g.edge_index = edge_index
                g.edge_attr = edge_attr

                start = i * Config.PARAMS.DATA[self.source]['N_INTERVALS'] + j
                end = start + window

                # switch from [F, N] (21, 228) -> [N, F] (228, 21)
                data_window = np.swapaxes(data[start:end, :], 0, 1) 

                # X feature vector for each node
                g.x = torch.FloatTensor(data_window[:, 0:Config.PARAMS.HYPER['N_HIST']])
                # Y ground truth for each node
                g.y = torch.FloatTensor(data_window[:, Config.PARAMS.HYPER['N_HIST']::])

                seqs += [g]
        
        # construct the actual dataset from sequence/collection of graph 
        self.save(seqs, self.processed_paths[0])

        # For PyG<2.4
        # data, slices = self.collate(seqs)
        # torch.save((data, slices), self.processed_paths[0])
        # https://pytorch-geometric.readthedocs.io/en/stable/tutorial/create_dataset.html

        name = Config.PARAMS.DATA[self.source]['NAME']
        n_nodes = Config.PARAMS.DATA[self.source]['N_NODES']
        
        torch.save((mean, std, name, n_nodes), self.processed_paths[1])