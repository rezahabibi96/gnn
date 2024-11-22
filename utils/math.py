import torch


def norm_z(x_raw, mean, std):
    """
    z-score normalization.

    :param x_raw: torch.array, input array to be normalized.
    :param mean: float, mean value of x_raw.
    :param std: float, std value of x_raw.

    :return: torch.array, normalized x_raw.
    """
    return (x_raw - mean)/std


def denorm_z(x_normed, mean, std):
    """
    z-score denormalization.

    :param x_norm: torch.array, input array to be denormalized.
    :param mean: float, mean value of x_normed.
    :param std: float, std value of x_normed.

    :return: torch.array, denormalized x_normed
    """
    return x_normed*std + mean


def calc_mae(y, y_hat):
    """
    calc mean absolute error.

    :param y: torch.array, truth value.
    :param y_hat: torch.array, pred value.

    :return: torch scalar, mae avg on all y_hat.
    """
    return torch.mean(torch.abs(y - y_hat))


def calc_rmse(y, y_hat):
    """
    calc root mean squared error.

    :param y: torch.array, truth value.
    :param y_hat: torch.array, pred value.

    :return: torch scalar, rmse avg on all y_hat.
    """
    return torch.sqrt(torch.mean((y - y_hat) ** 2))


def calc_mape(y, y_hat):
    """
    calc mean absolute percentage error.

    :param y: torch.array, truth value.
    :param y_hat: torch.array, pred value.

    :return: torch scala, mape avg on all y_hat.
    """
    
    return torch.mean(torch.abs(y - y_hat) / (y + 1e-15))