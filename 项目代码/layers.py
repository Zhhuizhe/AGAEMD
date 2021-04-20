import tensorflow as tf
from utils import *


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:

        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Encoder(Layer):
    """Encoder layer."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = name
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights1'] = weight_variable_glorot(
                input_dim, output_dim, name='weights1')
            self.vars['weights2'] = weight_variable_glorot(
                output_dim, output_dim, name='weights2')
            self.vars['weights3'] = weight_variable_glorot(
                output_dim, output_dim, name='weights3')
            self.vars['attention'] = tf.Variable(tf.constant([0.5, 0.33, 0.25]),name = 'attention')

        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        with tf.name_scope(self.name):
            x1 = inputs
            x1 = dropout_sparse(x1, 1-self.dropout, self.features_nonzero)  #dropout
            x1 = tf.sparse.sparse_dense_matmul(x1, self.vars['weights1'])
            x1 = tf.sparse.sparse_dense_matmul(self.adj, x1)
            outputs1 = self.act(x1)
            x2 = outputs1
            x2 = tf.nn.dropout(x2, 1-self.dropout)   # #dropout
            x2 = tf.matmul(x2, self.vars['weights2'])
            x2 = tf.sparse.sparse_dense_matmul(self.adj, x2)
            outputs2 = self.act(x2)
            x3 = outputs2
            x3 = tf.nn.dropout(x3, 1-self.dropout)
            x3 = tf.matmul(x3, self.vars['weights3'])
            x3 = tf.sparse.sparse_dense_matmul(self.adj, x3)
            outputs3 = self.act(x3)
            outputs = outputs1 * \
                              self.vars['attention'][0] +  outputs2 * self.vars['attention'][1] +  outputs3 * self.vars['attention'][2]
        return outputs


class Decoder(Layer):
    """Decoder layer."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights4'] = weight_variable_glorot(
                input_dim, input_dim, name='weights4')

    def _call(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 0.9-self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights4'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            # 该处将预测结果x由2d矩阵变为1d向量，并对结果使用激活函数
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs












