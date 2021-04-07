import numpy as np
import argparse
from utils import *

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--k_folds', default=5, type=int)

    dis_sim_mat = np.loadtxt("./HMDD32/disease_similarity_new2.txt", delimiter=' ')
    rna_sim_mat = np.loadtxt('./HMDD32/mir_fun_sim_matrix_new2.txt', delimiter=' ')
    rna_dis_adj_mat = np.loadtxt('./HMDD32/combine_association_matrix.txt', delimiter=' ')

    # 构建异构网络
    het_mat = construct_het_graph(rna_dis_adj_mat, rna_sim_mat, dis_sim_mat)
    A = construct_adj_net(rna_dis_adj_mat)

    # 5折交叉验证
    k_folds = 5
    mat_training = np.copy(rna_dis_adj_mat)
    mat_testing = np.copy(rna_dis_adj_mat)
    idx = np.array(np.where(rna_dis_adj_mat == 1)).T
    rng = np.random.default_rng()
    rng.shuffle(idx)
    idx = np.array_split(idx, 5)
    nsample = int(idx.shape[0])

    for i in range(k_folds):
        mat_training[idx[i]] = 0
        mat_testing = rna_dis_adj_mat - mat_training

    print(nsample)
