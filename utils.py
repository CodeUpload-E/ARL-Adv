import tensorflow as tf
import numpy as np


def inner_dot(a, b, axis=-1, keepdims=False, name=None):
    with tf.name_scope(name):
        return tf.reduce_sum(a * b, axis=axis, keepdims=keepdims)


def l2_norm_tensors(*tensors):
    return apply_tensors(lambda t: tf.nn.l2_normalize(t, axis=-1), *tensors)


def apply_tensors(func, *tensors):
    res = [func(t) for t in tensors]
    return res if len(res) > 1 else res[0]
