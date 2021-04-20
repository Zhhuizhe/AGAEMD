import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def load_data(rna_dis_adj, k_folds, dense=False):
    test_pos_list = list()
    test_pos_idx = np.array(np.where(rna_dis_adj == 1))
    test_neg_idx = np.array(np.where(rna_dis_adj == 0))
    rng = np.random.default_rng()
    rng.shuffle(test_pos_idx, axis=1)

    for test_pos in np.array_split(test_pos_idx, k_folds, axis=1):
        num_of_samples = len(test_pos[0])

        rng.shuffle(test_neg_idx, axis=1)
        test_neg = test_neg_idx[:, :num_of_samples]
        test_pos_list.append(np.hstack((test_pos, test_neg)))
    return test_pos_list


def normalize_mat(mat):
    assert mat.size != 0, f"Calculating normalized matrix need a non-zero square matrix. matrix size:{mat.shape}"
    mat_size = mat.shape[0]
    diag = np.zeros((mat_size, mat_size))
    np.fill_diagonal(diag, np.power(np.sum(mat, axis=0), -1/2))
    ret = diag.dot(mat).dot(diag)
    return ret


def construct_het_graph(rna_dis_mat, rna_mat, dis_mat, miu):
    # 计算相似度矩阵
    # rna_sim_mat - N_r * (N_r + N_d); dis_sim_mat - N_d * (N_r + N_d)
    rna_sim_mat = np.hstack((rna_mat * miu, rna_mat.dot(rna_dis_mat)))
    dis_sim_mat = np.hstack((dis_mat.dot(rna_dis_mat.T), dis_mat * miu))
    # N_r * (N_r + N_d) concat N_d * (N_r + N_d) -> (N_r + N_d) * (N_r + N_d)
    ret = np.vstack((rna_sim_mat, dis_sim_mat))
    return normalize_mat(ret)


def construct_adj_mat(rna_dis_mat):
    mat_tmp = rna_dis_mat.copy()
    # 二阶可达矩阵
    # mat_tmp = mat_tmp.dot(mat_tmp.T).dot(mat_tmp)
    mat_tmp[mat_tmp == 0] = -999
    rna_mat = np.zeros((rna_dis_mat.shape[0], rna_dis_mat.shape[0]))
    dis_mat = np.zeros((rna_dis_mat.shape[1], rna_dis_mat.shape[1]))

    mat1 = np.hstack((rna_mat, mat_tmp))
    mat2 = np.hstack((mat_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


# rna_dis_adj_mat, pred_adj_mat, training_mat类型为numpy.array
def calculate_auc(rna_dis_adj_mat, pred_adj_mat, training_mat):
    pred_adj_mat = np.reshape(pred_adj_mat, (713, 447))
    row_idx = training_mat[0]
    col_idx = training_mat[1]
    truth_score = rna_dis_adj_mat[row_idx, col_idx]
    pred_score = np.clip(pred_adj_mat[row_idx, col_idx], 0, 1)
    fpr, tpr, thresholds = roc_curve(truth_score, pred_score)
    ret = auc(fpr, tpr)
    return ret


def calculate_loss(pred, label, norm, pos_weight):
    return norm * F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight, reduction="mean")
