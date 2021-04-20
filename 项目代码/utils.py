import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from constant import *


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random.uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out*(1./keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def load_data(test_index):
    choice = VERSION
    if choice == 3.2:
        miRNA_dis_matrix = np.loadtxt('HMDD3.2/HMDDv3.2_from_Tensor_v2/combine_association_matrix.txt', delimiter=' ')
    if choice == 2:
        miRNA_dis_matrix = np.loadtxt('HMDD2.0data(383-495)/m-d.csv', delimiter=',')
    # 从disease-miRNA矩阵中找出为1的元素index
    index_matrix = np.mat(np.where(miRNA_dis_matrix == 1))  # 有1的元素下标 第一个元组为行下标，第二个元组为列下标
    association_nam = index_matrix.shape[1]

    train_matrix = np.mat(miRNA_dis_matrix).copy()  # 713 * 447

    # 将待测试元素的值置为0
    if CASE_STUDY == 0:
       train_matrix[tuple(np.array(test_index).T)] = 0   # 将一个subset变0（只剩下4个subsets作为train） (case study 时除外)
    else:
        pass

    # 矩阵转化为数组，并将2d数组转换为1d数组，长度为318711
    test_matrix = miRNA_dis_matrix - train_matrix
    logits_test = test_matrix.A
    logits_test = logits_test.reshape([-1, 1])

    logits_train = train_matrix.A
    logits_train = logits_train.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.float).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.float).reshape([-1, 1])

    # 降维
    logits_train = np.array([x for y in logits_train for x in y])
    logits_test = np.array([x for y in logits_test for x in y])

    return miRNA_dis_matrix, train_matrix, test_matrix, logits_train, logits_test, train_mask, test_mask


def construct_feed_dict(features, adj, adj_orig, dropout, adjdp, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['adjdp']: adjdp})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    return feed_dict


def masked_loss(preds, labels, num_u, num_v, association_nam):
    norm = num_u*num_v / float((num_u*num_v-association_nam) * 2)
    pos_weight = float(num_u*num_v-association_nam)/(association_nam)
    loss = norm * tf.reduce_mean(
       tf.nn.weighted_cross_entropy_with_logits(
           logits=preds,
           labels=labels,
           pos_weight=pos_weight))
    return loss
