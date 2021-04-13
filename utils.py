import torch
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def construct_het_graph(rna_dis_mat, rna_mat, dis_mat, miu):
    mat1 = np.hstack((rna_mat * miu, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat * miu))
    return np.vstack((mat1, mat2))


def construct_adj_mat(rna_dis_mat):
    drug_matrix = np.zeros((rna_dis_mat.shape[0], rna_dis_mat.shape[0]), dtype=np.int8)
    dis_matrix = np.zeros((rna_dis_mat.shape[1], rna_dis_mat.shape[1]), dtype=np.int8)

    mat1 = np.hstack((drug_matrix, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def calculate_auc(rna_dis_adj_mat, pred_adj_mat, training_mat):
    pred_adj_mat = torch.reshape(pred_adj_mat, (713, 447))
    idx = torch.where(training_mat == 0)
    row_idx = idx[0].long()
    col_idx = idx[1].long()
    truth_score = rna_dis_adj_mat[row_idx, col_idx]
    pred_score = pred_adj_mat[row_idx, col_idx].detach().numpy()
    fpr, tpr, thresholds = roc_curve(truth_score, pred_score)
    ret = auc(fpr, tpr)
    return ret
