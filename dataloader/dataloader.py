import numpy as np

from helpers import Config
from helpers import Log


def split_data(data, ratio):
    """
    given data, split it into subsets of train, val, and test.

    :param data: data to split.
    :param n_slots: possible number of sliding windows in a day.
    :param ratio: (train, val, test) ratio.

    :return train, val, test: splitted data.
    """
    r_train, r_val, _ = ratio

    n_train = Config.PARAMS.DATA[Config.PARAMS.ACTIVE_DATA]['N_SLOTS'] * r_train
    n_val = Config.PARAMS.DATA[Config.PARAMS.ACTIVE_DATA]['N_SLOTS'] * r_val

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test


def from_dist_to_weight(D, sigma2=0.1, epsilon=0.5, gat=False):
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
    if gat:
        W[W>0] = 1
        W += np.identity(n)
    
    return W