import time
import torch

from helpers import Config, Log
from runners import run_sl, run_cl, run_ol


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Config.PARAMS.CUDA['DEVICE'] = device 
    Log.info("using {}".format(Config.PARAMS.CUDA['DEVICE']))
    time_strf = time.strftime("%Y-%m-%d %H:%M:%S")

    if Config.PARAMS.ACTIVE['LEARNING'] == 'SL':
        run_sl(time_strf)


    elif Config.PARAMS.ACTIVE['LEARNING'] == 'CL':
        run_cl(time_strf)


    elif Config.PARAMS.ACTIVE['LEARNING'] == 'OL':
        run_ol(time_strf)


if __name__ == "__main__":
    main()