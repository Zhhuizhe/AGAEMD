import logging
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from numpy import inf

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import *
from model import AGAEMD

torch.set_default_tensor_type(torch.DoubleTensor)


def train(model, training_data, label, optimizer, pos_weight, norm):
    model.train()
    outputs = model(training_data)
    loss = calculate_loss(outputs, label, norm, pos_weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validation(model, validation_data):
    model.eval()


def train_agaemd():
    dis_sim_mat = np.loadtxt("./HMDD32/disease_similarity_new2.txt", delimiter=' ')
    rna_sim_mat = np.loadtxt('./HMDD32/mir_fun_sim_matrix_new2.txt', delimiter=' ')
    rna_dis_adj_mat = np.loadtxt('./HMDD32/combine_association_matrix.txt', delimiter=' ')

    # 设置模型参数
    args_config = {
        "num_heads_per_layer": [8, 16, 16, 32, 8, 3],
        "num_embedding_features": [512, 256, 256, 256, 128, 128],
        "num_hidden_layers": 6,
        "num_epoch": 10000,
        "dropout": 0.6,
        "attn_dropout": 0.4,
        "slope": 0.2,
        "mat_weight_coef": 0.8,
        "lr": 1e-3,  # 2e-4
        "weight_decay": 1e-4,
        "eval_freq": 200,
        "DENSE_OR_NOT": False,
        "LOAD_MODE": False
    }

    n_rna = rna_dis_adj_mat.shape[0]
    n_dis = rna_dis_adj_mat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 划分k折交叉验证数据集
    k_folds = 5
    testing_data_list = load_data(rna_dis_adj_mat, k_folds, args_config["DENSE_OR_NOT"])

    total_auc = 0
    auc_list = list()
    loss_list = list()

    for i in range(k_folds):
        print(f"********* {i + 1} of {k_folds}-flods *********")

        if args_config["LOAD_MODE"]:
            model = torch.load("./模型/AGAEMD_0.922.pth")
        else:
            # 创建模型
            model = AGAEMD(n_rna + n_dis,
                           args_config["num_hidden_layers"],
                           args_config["num_embedding_features"],
                           args_config["num_heads_per_layer"],
                           args_config["dropout"],
                           args_config["attn_dropout"],
                           args_config["slope"],
                           rna_sim_mat.shape[0],
                           dis_sim_mat.shape[0],
                           device).to(device)

        # 创建参数优化方案
        optimizer = Adam(model.parameters(), lr=args_config["lr"], weight_decay=args_config["weight_decay"])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

        # 构建异构网络
        training_mat = rna_dis_adj_mat.copy()
        training_mat[tuple(testing_data_list[i])] = 0
        n_pos_sample = np.sum(training_mat)

        # 构建异构矩阵和邻接矩阵
        het_mat = construct_het_graph(training_mat, rna_sim_mat, dis_sim_mat, args_config["mat_weight_coef"])
        adj_mat = construct_adj_mat(training_mat)
        pos_weight = torch.tensor((n_rna * n_dis - n_pos_sample) / n_pos_sample)
        norm = n_rna * n_dis / ((n_rna * n_dis - n_pos_sample) * 2)

        het_graph = torch.tensor(het_mat).to(device=device)
        adj_graph = torch.tensor(adj_mat).to(device=device)
        training_data = torch.reshape(torch.tensor(training_mat), (-1, )).to(device=device)
        graph_data = (het_graph, adj_graph)

        BEST_VAL_AUC, BEST_VAL_LOSS, PATIENCE_CNT = [0, inf, 0]

        for epoch in range(args_config["num_epoch"]):
            loss = train(model, graph_data, training_data, optimizer, pos_weight, norm)

            if args_config["eval_freq"] > 0 and (epoch == 0 or (epoch + 1) % args_config["eval_freq"] == 0):
                model.eval()
                link_pred = model(graph_data).cpu().detach().numpy()
                auc = calculate_auc(rna_dis_adj_mat, link_pred, testing_data_list[i])

                print(f"--------{epoch + 1} of {args_config['num_epoch']}--------")
                print(f"loss:{loss.item()}")
                print(f"auc:{auc}")
                auc_list.append(auc)
                loss_list.append(loss.item())

            if loss.item() < BEST_VAL_LOSS or auc > BEST_VAL_AUC:
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                BEST_VAL_AUC = max(auc, BEST_VAL_AUC)
                PATIENCE_CNT = 0
            else:
                PATIENCE_CNT += 1

            if PATIENCE_CNT > 1000:
                break

            scheduler.step()

        # 计算auc
        model.eval()
        link_pred = model(graph_data).cpu().detach().numpy()
        auc = calculate_auc(rna_dis_adj_mat, link_pred, testing_data_list[i])
        print(f"final AUC:{auc}")
        print(f"best AUC:{BEST_VAL_AUC}")
        total_auc += auc
        torch.save(model, f"AGAEMD_{i}.pth")

        plt.plot(np.arange(len(auc_list)), auc_list, linewidth=2.5)
        plt.plot(np.arange(len(loss_list)), loss_list, linewidth=2.5)
        plt.show()

    print("***********************")
    print(f"average auc:{total_auc / k_folds}")
    torch.save(model, "AGAEMD.pth")
    return


if __name__ == '__main__':
    train_agaemd()
