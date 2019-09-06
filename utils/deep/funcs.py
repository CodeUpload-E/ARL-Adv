import numpy as np
import tensorflow as tf

# import tensorflow.summary as su

i32 = tf.int32
i64 = tf.int64
f32 = tf.float32
f64 = tf.float64

relu = tf.nn.relu
tanh = tf.nn.tanh
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax


# histogram = su.histogram
# scalar = su.scalar


# tf.keras.layers.Dense()


def normal(shape, scale, loc=0):
    return np.random.normal(loc=loc, scale=scale, size=shape)


def parse_variable(value, trainable=True, name=None, dtype=tf.float32):
    if isinstance(value, tf.Variable) or isinstance(value, tf.Tensor):
        return value
    elif value is not None:
        return tf.Variable(value, trainable=trainable, name=name, dtype=dtype)


def inner_dot(a, b, axis=-1, keepdims=False, name=None):
    with tf.name_scope(name):
        return tf.reduce_sum(a * b, axis=axis, keepdims=keepdims)


def element_wise(a, b, func=tf.multiply, reduce=None, axis=None, keepdims=None, name=None):
    with tf.name_scope(name):
        wise = func(a, b)
        if reduce is not None:
            if callable(reduce):
                reduce_f = reduce
            elif isinstance(reduce, str):
                reduce_f = {'sum': tf.reduce_sum, 'mean': tf.reduce_mean}[reduce]
            else:
                raise TypeError('unusable reduce: {}'.format(reduce))
            wise = reduce_f(wise, axis=axis, keepdims=keepdims)
        return wise


def tensor_dot(t, w, b=None, axes=None, activation=None, name=None):
    with tf.name_scope(name):
        if axes is None:
            axes = ([-1], [0])
        res = tf.tensordot(t, w, axes)
        if b is not None:
            res = tf.add(res, b)
        if activation is not None:
            res = activation(res)
        return res


def tile_axes(t, axes, tile):
    rank = len(t.shape)
    multiplies = [1] * rank
    for a, n in zip(axes, tile):
        if n is not None and n > 1:
            multiplies[a if a >= 0 else rank + a] = n
    return tf.tile(t, multiplies) if sum(multiplies) > rank else t


def expand_dims(t, axes, tile=None):
    if isinstance(axes, (int, np.int)):
        t = tf.expand_dims(t, axis=axes)
        if tile is not None:
            t = tile_axes(t, axes=axes, tile=tile)
    else:
        axes = np.array(axes)
        axes[axes < 0] += len(t.shape)
        arg_sort = np.argsort(axes)
        axes = axes[arg_sort]
        for idx, axis in enumerate(axes):
            t = tf.expand_dims(t, axis=axis + idx)
        if tile is not None:
            axes = axes + np.arange(len(axes))
            tile = np.array(tile)[arg_sort]
            t = tile_axes(t, axes=axes, tile=tile)
    return t


def l2_norm_tensors(*tensors):
    return apply_tensors(lambda t: tf.nn.l2_normalize(t, axis=-1), *tensors)


def apply_tensors(func, *tensors):
    res = [func(t) for t in tensors]
    return res if len(res) > 1 else res[0]


def get_trainable(*objects):
    import utils.array_utils as au
    vs = au.merge([o] if isinstance(o, tf.Variable) else o.trainable_variables() for o in objects)
    return sort_variables(set([v for v in vs if v.trainable]))


def sort_variables(variables, key=lambda v: v.name):
    return sorted(variables, key=key)


def subtract_from(from_, *subs_):
    res = set(from_)
    for sub in subs_:
        res.difference_update(set(sub))
    return res


def get_train_op(lr, loss, var_list=None, opt_cls='Adam', name=None,
                 grads=False, grad_vars_func=None):
    from tensorflow.train import AdagradOptimizer, AdamOptimizer
    learning_rate = tf.train.exponential_decay(**lr) if isinstance(lr, dict) else lr
    optimizer_cls = {'adagrad': AdagradOptimizer, 'adam': AdamOptimizer}[opt_cls.lower()]
    optimizer = optimizer_cls(learning_rate)
    grad_vars = optimizer.compute_gradients(loss, var_list=var_list)
    if grad_vars_func is not None:
        grad_vars = grad_vars_func(grad_vars)
    train_op = optimizer.apply_gradients(grad_vars, name=name)
    return (train_op, grad_vars) if grads else train_op


def get_session(gpu_id, gpu_frac, allow_growth=False, run_init=True):
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_frac,
        allow_growth=allow_growth, visible_device_list=str(gpu_id)
    )
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    if run_init:
        init_session(sess)
    return sess


def init_session(sess):
    sess.run(tf.global_variables_initializer())


def add_dropout(tensor, keep_prob, name=None):
    return tf.nn.dropout(tensor, keep_prob=keep_prob, name=name)


def add_batch_norm(tensor, is_train, name=None):
    return tf.layers.batch_normalization(tensor, training=is_train, name=name)


def wrap_batch_norm(is_train):
    def add_bn(*args): return add_batch_norm(args[0], is_train)

    return add_bn


def control_dependencies(cond, train_func):
    if cond:
        update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_func()
    else:
        train_func()
