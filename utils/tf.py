import os
import glob
import traceback
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def from_tfevent_to_pandas(path):
    """
    convert tfevent log to pandas df.
    https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python.

    :param path: given path (tensorboard logs).

    :return df: pandas df.
    """
    path_tfevent = os.path.join('./artifacts/tensorboard', f'{path}/*')
    path_pandas = os.path.join('./artifacts/metrics', f'{path}')

    Path(path_tfevent).mkdir(parents=True, exist_ok=True)
    Path(path_pandas).mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({"metric": [], "value": [], "step": [], "name": [], "wall_time": []})

    for file in glob.glob(path_tfevent):
        try:
            event_accumulator = EventAccumulator(file, size_guidance={"scalars": 0})
            event_accumulator.Reload()

            tags = event_accumulator.Tags()["scalars"]
            for tag in tags:
                event_list = event_accumulator.Scalars(tag)            
                value = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                wall_time = list(map(lambda x: x.wall_time, event_list))
                
                runlog = {"metric": [tag] * len(step), "value": value, "step": step, 
                          "name": [file.split('.')[-1]] * len(step), "wall_time": wall_time}
                runlog = pd.DataFrame(runlog)
                
                data = pd.concat([data, runlog])
        
        except Exception:
            print("event file possibly corrupt: {}".format(file))
            traceback.print_exc()

    data.to_csv(f'{path_pandas}/csv.csv')
    return data


if __name__ == "__main__":
    # python3 -m utils.tf
    path = '2024-12-04 14:31:28'
    from_tfevent_to_pandas(path)