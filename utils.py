import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def normalize_mat(mat):
    assert mat.size != 0, f"Calculating normalized matrix need a non-zero square matrix. matrix size:{mat.shape}"
    mat_size = mat.shape[0]
    diag = np.zeros((mat_size, mat_size))
    np.fill_diagonal(diag, np.power(np.sum(mat, axis=0), -1/2))
    ret = diag.dot(mat).dot(diag)
    return ret


def construct_het_graph(rna_dis_mat, rna_mat, dis_mat, miu):
    # rna_mat = normalize_mat(rna_mat)
    # dis_mat = normalize_mat(dis_mat)
    mat1 = np.hstack((rna_mat * miu, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat * miu))
    ret = np.vstack((mat1, mat2))
    # ret[ret == 0] = -10000
    return ret


def construct_adj_mat(rna_dis_mat):
    mat_tmp = rna_dis_mat.copy()
    mat_tmp[mat_tmp == 0] = -50
    rna_mat = np.zeros((rna_dis_mat.shape[0], rna_dis_mat.shape[0]))
    dis_mat = np.zeros((rna_dis_mat.shape[1], rna_dis_mat.shape[1]))

    mat1 = np.hstack((rna_mat, mat_tmp))
    mat2 = np.hstack((mat_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    # ret[ret == 0] = -10000
    return ret


# rna_dis_adj_mat, pred_adj_mat, training_mat类型为numpy.array
def calculate_auc(rna_dis_adj_mat, pred_adj_mat, training_mat):
    pred_adj_mat = np.reshape(pred_adj_mat, (713, 447))
    idx = np.where(training_mat == 0)
    truth_score = rna_dis_adj_mat[idx]
    pred_score = pred_adj_mat[idx]
    fpr, tpr, thresholds = roc_curve(truth_score, pred_score)
    ret = auc(fpr, tpr)
    return ret


def calculate_loss(pred, label, norm, pos_weight):
    return norm * F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight, reduction="mean")
