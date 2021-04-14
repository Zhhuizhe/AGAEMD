import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import construct_het_graph
from utils import construct_adj_mat
from utils import calculate_auc
from utils import calculate_loss
from model import AGAEMD

torch.set_default_tensor_type(torch.DoubleTensor)


def loss_func(inputs, labels):
    return


if __name__ == '__main__':
    dis_sim_mat = np.loadtxt("./HMDD32/disease_similarity_new2.txt", delimiter=' ')
    rna_sim_mat = np.loadtxt('./HMDD32/mir_fun_sim_matrix_new2.txt', delimiter=' ')
    rna_dis_adj_mat = np.loadtxt('./HMDD32/combine_association_matrix.txt', delimiter=' ')

    # 设置模型参数
    args_config = {
        "num_heads_per_layer": [3, 8, 8],
        "num_embedding_features": [256, 256, 256],
        "num_hidden_layers": 2,
        "num_epoch": 200,
        "dropout": 0.6,
        "slope": 0.2,
        "mat_weight_coef": 0.8,
        "lr": 1e-3,
        "weight_decay": 5e-4
    }

    # k折交叉验证
    k_folds = 10
    training_mat = np.copy(rna_dis_adj_mat)
    testing_mat = np.copy(rna_dis_adj_mat)
    idx = np.array(np.where(rna_dis_adj_mat == 1)).T
    np.random.shuffle(idx)
    idx = np.array_split(idx, k_folds)

    n_rna = rna_dis_adj_mat.shape[0]
    n_dis = rna_dis_adj_mat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AGAEMD(n_rna + n_dis,
                   args_config["num_hidden_layers"],
                   args_config["num_embedding_features"],
                   args_config["num_heads_per_layer"],
                   args_config["dropout"],
                   args_config["slope"],
                   rna_sim_mat.shape[0],
                   dis_sim_mat.shape[0]).to(device)
    """
    model = AGAEMD(rna_dis_adj_mat.shape[0] + rna_dis_adj_mat.shape[1],
                   gat_config["num_embedding_features"],
                   gat_config["dropout"],
                   gat_config["slope"],
                   gat_config["num_heads_per_layer"],
                   rna_sim_mat.shape[0], dis_sim_mat.shape[0])
    """
    optimizer = Adam(model.parameters(), lr=args_config["lr"], weight_decay=args_config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    for i in range(k_folds):
        print(f"********* {i + 1} of {k_folds}-flods *********")
        training_mat[idx[i]] = 0
        n_pos_sample = np.sum(training_mat)
        # testing_mat = rna_dis_adj_mat - training_mat

        # 构建异构网络
        het_mat = construct_het_graph(training_mat, rna_sim_mat, dis_sim_mat, args_config["mat_weight_coef"])
        adj_mat = construct_adj_mat(training_mat)
        pos_weight = torch.tensor((n_rna * n_dis - n_pos_sample) / n_pos_sample)
        norm = n_rna * n_dis / ((n_rna * n_dis - n_pos_sample) * 2)

        het_graph = torch.tensor(het_mat).to(device=device)
        adj_graph = torch.tensor(adj_mat).to(device=device)
        training_data = torch.tensor(training_mat).to(device=device)
        graph_data = (het_graph, adj_graph)

        for epoch in range(args_config["num_epoch"]):
            model.train()

            outputs = model(graph_data)
            truth_label = torch.reshape(training_data, (-1,))
            loss = calculate_loss(outputs, truth_label, norm, pos_weight)
            # loss = loss_fn(pred_label, truth_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"loss:{loss.item()}")

        # 计算auc
        link_pred = model(graph_data).cpu().detach().numpy()
        auc = calculate_auc(rna_dis_adj_mat, link_pred, training_mat)
        print(auc)


