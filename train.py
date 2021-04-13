import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import construct_het_graph
from utils import construct_adj_mat
from utils import calculate_auc
from model import AGAEMD
from constant import *

torch.set_default_tensor_type(torch.DoubleTensor)


def loss_func(inputs, labels):
    return


if __name__ == '__main__':
    dis_sim_mat = np.loadtxt("./HMDD32/disease_similarity_new2.txt", delimiter=' ')
    rna_sim_mat = np.loadtxt('./HMDD32/mir_fun_sim_matrix_new2.txt', delimiter=' ')
    rna_dis_adj_mat = np.loadtxt('./HMDD32/combine_association_matrix.txt', delimiter=' ')

    gat_config = {
        "num_heads_per_layer": [8, 8, 3],
        "num_embedding_features": [256, 256, 256],
        "dropout": 0.6,
        "slope": 0.2
    }

    # 5折交叉验证
    k_folds = 10
    training_mat = np.copy(rna_dis_adj_mat)
    testing_mat = np.copy(rna_dis_adj_mat)
    idx = np.array(np.where(rna_dis_adj_mat == 1)).T
    rng = np.random.default_rng()
    rng.shuffle(idx)
    idx = np.array_split(idx, k_folds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AGAEMD(rna_dis_adj_mat.shape[0] + rna_dis_adj_mat.shape[1],
                   gat_config["num_embedding_features"],
                   gat_config["dropout"],
                   gat_config["slope"],
                   gat_config["num_heads_per_layer"],
                   rna_sim_mat.shape[0], dis_sim_mat.shape[0]).to(device)
    """
    model = AGAEMD(rna_dis_adj_mat.shape[0] + rna_dis_adj_mat.shape[1],
                   gat_config["num_embedding_features"],
                   gat_config["dropout"],
                   gat_config["slope"],
                   gat_config["num_heads_per_layer"],
                   rna_sim_mat.shape[0], dis_sim_mat.shape[0])
    """
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_fn = nn.MSELoss(reduction='mean')

    #
    for i in range(k_folds):
        print(f"********* {i + 1} of {k_folds}-flods *********")
        training_mat[idx[i]] = 0
        # testing_mat = rna_dis_adj_mat - training_mat

        # 构建异构网络
        het_mat = construct_het_graph(training_mat, rna_sim_mat, dis_sim_mat, MAT_WEIGHT_COEF)
        adj_mat = construct_adj_mat(training_mat)

        het_graph = torch.tensor(het_mat).to(device=device)
        adj_graph = torch.tensor(adj_mat).to(device=device)
        training_data = torch.tensor(training_mat).to(device=device)
        testing_data = torch.tensor(testing_mat).to(device=device)

        for epoch in range(NUM_OF_EPOCH):
            optimizer.zero_grad()

            outputs = model(adj_graph, het_graph)
            target = torch.reshape(training_data, (-1,))
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"loss:{loss.item()}")

        # 计算auc
        link_pred = model(adj_graph, het_graph).cpu().detach().numpy()
        auc = calculate_auc(rna_dis_adj_mat, link_pred, training_mat)
        print(auc)


