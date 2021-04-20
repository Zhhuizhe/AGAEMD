from layers import *

import tensorflow as tf


from utils import *

from constant import  *
from clr import *

learning_rate  = LEARNING_RATE
lr      =   LEARNING_RATE


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.hid = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        # activations
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]
        self.hid = self.activations[-2]

        # Store model variables for easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics+
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step,)

    def predict(self):
        pass

    def hidd(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GAutoencoder(Model, ):
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, num_c,association_nam, name,
                 act=tf.nn.elu, **kwargs):  # tf.nn.elu
        super(GAutoencoder, self).__init__(**kwargs)
        print("model :placeholders:")
        print(placeholders)
        self.placeholders = placeholders
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features   # 输入矩阵的特征维数
        self.emb_dim = emb_dim          # embedding维数
        self.features_nonzero = features_nonzero  # 20056
        self.adj_nonzero = adj_nonzero    # 564612
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act  # 激活函数为elu函数
        self.global_step = tf.Variable(0, trainable=False)
        # 使用Adam优化各变量
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=cyclic_learning_rate(global_step=self.global_step, learning_rate=lr*0.1, max_lr=lr, mode='exp_range', gamma=.995))
        #
        self.num_r = num_r
        self.num_c = num_c
        self.association_num = association_nam
        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def _loss(self):
        self.loss = masked_loss(self.outputs, self.placeholders['labels'], self.num_r, self.num_c,
                                     self.association_num)

    def _accuracy(self):
        #         self.accuracy = euclidean_loss(self.outputs, self.placeholders['labels'])
        self.accuracy = masked_loss(self.outputs, self.placeholders['labels'], self.num_r, self.num_c,
                                     self.association_num)

    def _build(self):

        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        self.layers.append(Encoder(
                                    name='Encoder',
                                    input_dim=self.input_dim,
                                    output_dim=self.emb_dim,
                                    adj=self.adj,
                                    features_nonzero=self.features_nonzero,
                                    dropout=self.dropout,
                                    act=self.act)
        )

        self.layers.append(Decoder(input_dim=self.emb_dim, name='Decoder', num_r = self.num_r, dropout=0., act=tf.nn.sigmoid))

    def predict(self):
        return self.outputs