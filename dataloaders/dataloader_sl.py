from helpers import Config


def split_data(data, ratio):
    """
    given data, split it into subsets of train, val, and test.

    :param data: data to split.
    :param ratio: (train, val, test) ratio specified by number of days.

    :return train, val, test: splitted data.
    """
    name = data.name

    r_train, r_val, _ = ratio

    n_train = Config.PARAMS.DATA[name]['N_SLOTS'] * r_train
    n_val = Config.PARAMS.DATA[name]['N_SLOTS'] * r_val

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test