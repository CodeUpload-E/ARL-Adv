from .funcs import *
from utils import au


class Dense:
    zero_initer = tf.zeros_initializer(dtype=f32)
    normal_initer = tf.random_normal_initializer(mean=0., stddev=1., dtype=f32)

    def __init__(self, in_dim, out_dim, activation=None, kernel=None, bias=None, name=None):
        self.name = name
        self.activation = activation
        if kernel is None:
            kernel = self.normal_initer
        if bias is None:
            bias = self.zero_initer
        with tf.variable_scope(name):
            w_name, w_shape, w_init = 'w', [in_dim, out_dim], kernel
            b_name, b_shape, b_init = 'b', [out_dim, ], bias
            self.w = tf.get_variable(name=w_name, shape=w_shape, initializer=w_init)
            self.b = tf.get_variable(name=b_name, shape=b_shape, initializer=b_init)

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, tensor, pre_apply=None, post_apply=None, name=None):
        with tf.name_scope(name):
            if callable(pre_apply):
                tensor = pre_apply(tensor)
            rank = len(tensor.shape)
            if rank > 2:
                output = tf.tensordot(tensor, self.w, [[rank - 1], [0]])
            else:
                output = tf.matmul(tensor, self.w)
            output = tf.nn.bias_add(output, self.b)
            if self.activation is not None:
                output = self.activation(output)
            if callable(post_apply):
                output = post_apply(output)
        return output

    def get_norm(self, order, add_bias=False):
        norm = tf.norm(self.w, ord=order)
        if add_bias:
            norm += tf.norm(self.b, ord=order)
        return norm

    def trainable_variables(self):
        return [self.w, self.b]


class Denses:
    def __init__(self, in_dim, out_dim_acts, kernel=None, bias=None, name=None):
        acts = [a for d, a in out_dim_acts]
        dims = [in_dim] + [d for d, a in out_dim_acts]
        in_out_dim_acts = [(dims[i], dims[i + 1], acts[i]) for i in range(len(acts))]
        with tf.variable_scope(name):
            self.denses = [Dense(
                in_dim=ind, out_dim=oud, kernel=kernel, bias=bias,
                activation=act, name='layer_{}'.format(idx))
                for idx, (ind, oud, act) in enumerate(in_out_dim_acts)]

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, tensor, name=None, pre_apply=None, post_apply=None, last=True):
        res = [tensor]
        with tf.name_scope(name):
            for i, d in enumerate(self.denses):
                a = res[-1]
                if callable(pre_apply):
                    a = pre_apply(a, i)
                a = d.apply(a, name='apply_{}'.format(i))
                if callable(post_apply):
                    a = post_apply(a, i)
                res.append(a)
        res.pop(0)
        return res[-1] if last else res

    def get_norm(self, order, add_bias=False):
        return sum(d.get_norm(order=order, add_bias=add_bias) for d in self.denses)

    def trainable_variables(self):
        return au.merge(d.trainable_variables() for d in self.denses)


class MyDense:
    def __init__(self, w_init=None, b_init=None, activation=None, name=None):
        self.w = parse_variable(w_init, name='{}_w'.format(name) if name else None)
        self.b = parse_variable(b_init, name='{}_b'.format(name) if name else None)
        self.activation = activation

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, t, axes=None, dropout=None, l2norm=False, name=None):
        t = tensor_dot(t, self.w, self.b, axes=axes, activation=self.activation, name=name)
        t = tf.nn.dropout(t, dropout) if dropout is not None else t
        t = l2_norm_tensors(t) if l2norm else t
        return t

    def trainable_variables(self):
        return [self.w, self.b]

    def get_norm(self, order, add_bias=True):
        norm = tf.norm(self.w, ord=order)
        if add_bias and self.b is not None:
            norm += tf.norm(self.b, ord=order)
        return norm

    @staticmethod
    def normal(in_dim, out_dim, scale, activation=None, name=None):
        return MyDense(w_init=normal((in_dim, out_dim), scale),
                       b_init=normal((out_dim,), scale),
                       activation=activation, name=name)


class MyDenses:
    def __init__(self, in_dim, out_dims, scale, activations=None, name=None):
        dims = (in_dim, *out_dims)
        self.denses = [MyDense.normal(
            in_dim=dims[i], out_dim=dims[i + 1], scale=scale,
            activation=self.try_index(activations, i),
            name='{}_layer{}'.format(name, i) if name else None)
            for i in range(len(dims) - 1)]

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def try_index(self, o, i):
        from collections import Sequence
        if isinstance(o, Sequence):
            return o[i]
        else:
            return o

    def apply(self, t, dropout=None, l2norm=None, last=True):
        res = [t]
        for i in range(len(self.denses)):
            do = self.try_index(dropout, i)
            l2 = self.try_index(l2norm, i)
            out_i = self.denses[i].apply(res[-1], dropout=do, l2norm=l2)
            res.append(out_i)
        res.pop(0)
        return res[-1] if last else res

    def trainable_variables(self):
        return au.merge(d.trainable_variables() for d in self.denses)

    def get_norm(self, order, add_bias=True):
        return sum(d.get_norm(order, add_bias) for d in self.denses)


if __name__ == '__main__':
    print(expand_dims(tf.Variable(np.random.normal(size=[2, 3, 4])), [1, 2, 2], [6, 5, 7]).shape)
    print(expand_dims(tf.Variable(np.random.normal(size=[2, 3, 4, 5])), 2, 6).shape)
