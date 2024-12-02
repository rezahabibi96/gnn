from helpers import Config
from dataprocessors import TrafficDataset


def create_dataset():
    """
    func to create dataset based on active data

    :return dataset: created dataset.
    """
    if Config.PARAMS.ACTIVE['DATA'] == 'PEMSD7':
        dataset = TrafficDataset()

    return dataset