from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from model import GAutoencoder
from constant import *

tf.compat.v1.disable_eager_execution()


def train(test_index, miRNA_matrix, dis_matrix):
    # Settings
    epochs = EPOCH
    dropout = DROUPOUT
    adjdp = ADJ_DP
    # 加载数据
    # miRNA_dis_matrix - miRNA-disease邻接矩阵; logits_train, logits_test - 矩阵变为1维向量结果(1*318711)
    # train_mask, test_mask - 1维向量转置结果(318711*1)
    miRNA_dis_matrix, train_matrix, test_matrix, logits_train, logits_test, train_mask, test_mask = load_data(test_index)

    association_nam = train_matrix.sum()
    num_r = train_matrix.shape[0]
    num_c = train_matrix.shape[1]

    # np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    # tf.set_random_seed(seed).

    # 创建异构网络矩阵G，并生成对应的压缩矩阵
    adj = constructHNet(train_matrix, miRNA_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    # 创建论文临界矩阵A
    X = constructNet(train_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    # 标准化异构网络矩阵G
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]

    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=()),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(logits_train.size,)),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'negative_mask': tf.compat.v1.placeholder(tf.int32)
    }
    emb_dim = EMB_DIM
    # 创建图自编码模型，该模型中也已经包括了解码器
    model = GAutoencoder(placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, num_c, association_nam, name='LAGCN')    # Initialize session
    sess = tf.compat.v1.Session()

    # Define model evaluation function
    def evaluate(features, adj, adj_orig, dropout, adjdp, lables, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, adj, adj_orig, dropout, adjdp, lables, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # 初始化变量
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):

        t = time.time()
        feed_dict = construct_feed_dict(features, adj_norm, adj_orig, dropout, adjdp, logits_train, train_mask, placeholders)
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            # print("Epoch:", '%04d' % (epoch + 1),
            #             #       "train_loss=", "{:.5f}".format(outs[1]))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    outs = sess.run(model.outputs, feed_dict=feed_dict)
    sess.close()
    return miRNA_dis_matrix, outs, train_matrix
