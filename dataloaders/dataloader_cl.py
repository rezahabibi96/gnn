from helpers import Config

def get_joint(data):
    """
    Given data, get joint data for training CL (upper bound).
    The joint training data comes from each context training data.

    :param data: whole data.

    :return joint: joint data for training.
    """
    name = data.name
    n_slots = Config.PARAMS.DATA[name]['N_SLOTS']

    join_train_ixs = []    
    ctx_obj = Config.PARAMS.DATA[name]['CONTEXT']

    for detail in ctx_obj['DETAIL']:

        # detail['TRAIN_START'] spcecifies starting day of context training data
        # train_st_ix specifies starting index (after being multiplied by n_slots)
        train_st_ix = (detail['TRAIN_START'] - 1) * n_slots
        train_en_ix = detail['TRAIN_END'] * n_slots

        # join_train_ixs defines the joint indices of training data for joint training (upper bound) 
        join_train_ixs = join_train_ixs + list(range(train_st_ix, train_en_ix))

    joint = data.index_select(join_train_ixs)
    return joint


def get_ctxs(data):
    """
    Given data, split it into contexts for CL set up.
    Each context consists of (train, val, test).

    :param data: whole data to split.

    :return ctxs: contexts objects.
    """
    name = data.name
    n_slots = Config.PARAMS.DATA[name]['N_SLOTS']
    
    ctxs = []
    ctx_obj = Config.PARAMS.DATA[name]['CONTEXT']

    for detail in ctx_obj['DETAIL']:
        id = detail['ID']

        train_st_ix = (detail['TRAIN_START'] - 1) * n_slots
        train_en_ix = detail['TRAIN_END'] * n_slots

        test_st_ix = (detail['TEST_START'] - 1) * n_slots
        test_en_ix = detail['TEST_END'] * n_slots

        train = data[train_st_ix:train_en_ix]
        test = data[test_st_ix:test_en_ix]

        ctx = {
            'id': id,
            'train': train,
            'test': test
        }

        ctxs.append(ctx)

    return ctxs