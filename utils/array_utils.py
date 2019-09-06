from collections import Counter, OrderedDict as Od

import numpy as np
from sklearn import metrics

eval_scores = ('acc', 'ari', 'nmi')


def merge(array):
    res = list()
    for a in array:
        res.extend(a)
    return res


def group_data_frame(data_frame, column):
    value_set = sorted(set(data_frame[column]))
    return [(Od([(column, v)]), data_frame[data_frame[column] == v]) for v in value_set]


def group_data_frame_columns(data_frame, columns):
    groups = [(Od(), data_frame)]
    for col in columns:
        for _ in range(len(groups)):
            p_od, p_df = groups.pop(0)
            for n_od, n_df in group_data_frame(p_df, col):
                p_od = p_od.copy()
                p_od.update(n_od)
                groups.append((p_od, n_df))
    return groups


def shuffle(array, inplace=False):
    array = array if inplace else array[:]
    np.random.shuffle(array)
    return array


def rehash(items, sort=True):
    items = sorted(set(items)) if sort else set(items)
    return {item: idx for idx, item in enumerate(items)}


def reindex(array):
    item2idx = dict((item, idx) for idx, item in enumerate(sorted(set(array))))
    return [item2idx[item] for item in array]


def count_occurence(y1, y2):
    y1_to_counter = Od((y, Counter()) for y in set(y1))
    for v1, v2 in zip(y1, y2):
        y1_to_counter[v1][v2] += 1
    return y1_to_counter


def score(y_true, y_pred, using_score):
    func = {
        'acc': acc,
        'ari': metrics.adjusted_rand_score,
        'auc': metrics.roc_auc_score,
        'nmi': metrics.normalized_mutual_info_score,
    }[using_score.lower()]
    return round(float(func(y_true, y_pred)), 4)


def scores(y_true, y_pred, using_scores=eval_scores):
    return Od((s, score(y_true, y_pred, s)) for s in using_scores)


def acc(y_true, y_pred):
    from sklearn.utils.linear_assignment_ import linear_assignment
    y_true, y_pred = reindex(y_true), reindex(y_pred)
    assert len(y_true) == len(y_pred)
    d = max(max(y_true), max(y_pred)) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for y_t, y_p in zip(y_true, y_pred):
        w[y_t][y_p] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i][j] for i, j in ind]) / len(y_pred)


def mean_std(array):
    return np.mean(array), np.std(array, ddof=1)


def permutation_generator(v_max, generator=None):
    if 0 in v_max:
        raise ValueError('0 should not appear in v2max')
    v_num = len(v_max)
    idx_vec = [0] * v_num
    while True:
        yield generator(idx_vec) if generator else idx_vec
        idx = 0
        while idx < v_num and idx_vec[idx] == v_max[idx] - 1:
            idx_vec[idx] = 0
            idx += 1
        if idx < v_num:
            idx_vec[idx] += 1
        else:
            return


def grid_params(name_value_list):
    def vec2dict(idx_vec):
        od = Od(zip(names, [None] * len(names)))
        for n_idx, v_idx in enumerate(idx_vec):
            n, v = names[n_idx], values_list[n_idx][v_idx]
            if not callable(v):
                od[n] = v
        return od

    names, values_list = list(zip(*name_value_list))
    v_len = [len(v) for v in values_list]
    return [od for od in permutation_generator(v_len, vec2dict)]


def cosine_similarity(mtx1, mtx2=None, dense_output=True):
    return metrics.pairwise.cosine_similarity(mtx1, mtx2, dense_output)


def transpose(array):
    # items in array should share the same length
    return list(zip(*array))


def split_multi_process(array, p_num):
    import math
    return list(split_slices(array, math.ceil(len(array) / p_num)))


def split_slices(array, batch_size):
    for since, until in split_since_until(len(array), batch_size):
        yield array[since: until]


def split_since_until(max_len, batch_size):
    since, until = 0, min(max_len, batch_size)
    while since < max_len:
        yield since, until
        since += batch_size
        until += min(max_len - until, batch_size)


def _can_include(v, include=None, exclude=None):
    if include is not None:
        return v in include
    if exclude is not None:
        return v not in exclude
    return True


def entries2name(entries, include=None, exclude=None, inner='=', inter=',', postfix=''):
    from collections import Iterable
    if isinstance(entries, dict):
        kv_list = entries.items()
    elif isinstance(entries, Iterable):
        kv_list = entries
    else:
        raise TypeError('unexpected type : {}'.format(type(entries)))
    pairs = ['{}{}{}'.format(k, inner, v) for k, v in kv_list if _can_include(k, include, exclude)]
    return inter.join(pairs) + postfix


def name2entries(name, include=None, exclude=None, inner='=', inter=',', postfix=''):
    if not isinstance(name, str):
        raise TypeError('unexpected type : {}'.format(type(name)))
    kv_list = [kv_pair.split(inner) for kv_pair in name.rstrip(postfix).split(inter)]
    entries = [(k, v) for k, v in kv_list if _can_include(k, include, exclude)]
    return entries
