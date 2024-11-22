import torch


def norm_z(x_raw, mean, std):
    """
    z-score normalization.

    :param x_raw: torch.array, input array to be normalized.
    :param mean: float, mean value of x_raw.
    :param std: float, std value of x_raw.

    :return torch.array, normalized x_raw.
    """
    return (x_raw - mean)/std


def denorm_z(x_normed, mean, std):
    """
    z-score denormalization.

    :param x_norm: torch.array, input array to be denormalized.
    :param mean: float, mean value of x_normed.
    :param std: float, std value of x_normed.

    :return torch.array, denormalized x_normed
    """
    return x_normed*std + mean