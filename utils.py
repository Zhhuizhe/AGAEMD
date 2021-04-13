import torch
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
    rna_mat = normalize_mat(rna_mat)
    dis_mat = normalize_mat(dis_mat)
    mat1 = np.hstack((rna_mat * miu, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat * miu))
    ret = np.vstack((mat1, mat2))
    ret[ret == 0] = -999
    return ret


def construct_adj_mat(rna_dis_mat):
    drug_matrix = np.zeros((rna_dis_mat.shape[0], rna_dis_mat.shape[0]), dtype=np.int8)
    dis_matrix = np.zeros((rna_dis_mat.shape[1], rna_dis_mat.shape[1]), dtype=np.int8)

    mat1 = np.hstack((drug_matrix, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def calculate_auc(rna_dis_adj_mat, pred_adj_mat, training_mat):
    pred_adj_mat = np.reshape(pred_adj_mat, (713, 447))
    idx = np.where(training_mat == 0)
    row_idx = idx[0]
    col_idx = idx[1]
    truth_score = rna_dis_adj_mat[row_idx, col_idx]
    pred_score = pred_adj_mat[row_idx, col_idx]
    print(np.max(pred_score))
    # pred_score[pred_score > 0.5] = 1
    # pred_score[pred_score <= 0.5] = 0
    fpr, tpr, thresholds = roc_curve(truth_score, pred_score)
    ret = auc(fpr, tpr)
    return ret
