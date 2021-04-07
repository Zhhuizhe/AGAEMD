import numpy as np


def construct_het_graph(rna_dis_mat, rna_mat, dis_mat):
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    return np.vstack(mat1, mat2)


def construct_adj_net(rna_dis_mat):
    drug_matrix = np.zeros((rna_dis_mat.shape[0], rna_dis_mat.shape[0]), dtype=np.int8)
    dis_matrix = np.zeros((rna_dis_mat.shape[1], rna_dis_mat.shape[1]), dtype=np.int8)

    mat1 = np.hstack((drug_matrix, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj
