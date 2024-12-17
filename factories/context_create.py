from helpers import Config


def create_ctxs(data, len_train, len_test, len_bufs):
    """
    Given data, create contexts for CL set up.
    Each context consists of (train, val, test).

    :param data: whole data to split.

    :return ctxs: contexts objects.
    """
    name = data.name

    n_days = Config.PARAMS.DATA[name]['N_DAYS']
    n_slots = Config.PARAMS.DATA[name]['N_SLOTS']

    len_ctxs = int( n_days / (len_train + len_test + len_bufs) )
    index = 0
    
    ctxs = []
    for id in range(len_ctxs):
        train_st_ix = index * n_slots
        train_en_ix = (index + len_train) * n_slots

        test_st_ix = (index + len_train) * n_slots
        test_en_ix = (index + len_train + len_test) * n_slots

        train = data[train_st_ix:train_en_ix]
        test = data[test_st_ix:test_en_ix]

        ctx = {
            'id': id,
            'train': train,
            'test': test
        }

        index = index + len_train + len_test

        ctxs.append(ctx)

    return ctxs