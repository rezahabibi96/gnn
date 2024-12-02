import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def from_tfevent_to_pandas(path):
    """
    convert tfevent log to pandas df.
    https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python.

    :param path: given path (tensorboard logs).

    :return df: pandas df.
    """
    data = pd.DataFrame({"event_metric": [], "value": [], "step": [], "wall_time": []})
    
    try:
        event_accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
        event_accumulator.Reload()

        tags = event_accumulator.Tags()["scalars"]
        for tag in tags:
            event_list = event_accumulator.Scalars(tag)            
            value = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            wall_time = list(map(lambda x: x.wall_time, event_list))
            
            runlog = {"event_metric": [tag] * len(step), "value": value, "step": step, "wall_time": wall_time}
            runlog = pd.DataFrame(runlog)
            
            data = pd.concat([data, runlog])
    
    except Exception:
        print("event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    
    return data


def from_many_tfevent_to_pandas(path):
    """
    convert many tfevent log to pandas df.
    https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python.

    :param path: given path (tensorboard logs).

    :return df: pandas df.
    """
    pass


if __name__ == "__main__":
    # python3 -m utils.tf
    from_tfevent_to_pandas('./artifacts/tensorboard').to_csv('csv.csv')