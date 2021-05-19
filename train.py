from numpy import inf

import torch
from torch.optim import Adam

from utils import *
from model import AGAEMD

torch.set_default_tensor_type(torch.DoubleTensor)


def train_agaemd(version="hdmm2"):
    if version.lower() == "hdmm2":
        dis_sim_mat = np.loadtxt("./HMDD2/d-d(diag_zero).csv", delimiter=' ')
        rna_sim_mat = np.loadtxt('./HMDD2/m-m_diag_zero.csv', delimiter=' ')
        rna_dis_adj_mat = np.loadtxt('./HMDD2/m-d.csv', delimiter=',')
    else:
        dis_sim_mat = np.loadtxt("./HMDD32/disease_similarity_new2.txt", delimiter=' ')
        rna_sim_mat = np.loadtxt('./HMDD32/mir_fun_sim_matrix_new2.txt', delimiter=' ')
        rna_dis_adj_mat = np.loadtxt('./HMDD32/combine_association_matrix.txt', delimiter=' ')

    # 设置模型参数
    args_config = {
        "num_heads_per_layer": [16, 32, 32, 16],
        "num_embedding_features": [512, 512, 512, 512],
        "num_hidden_layers": 4,
        "num_epoch": 500,
        "slope": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "penalty_factor": 0.5,
        "eval_freq": 50
    }

    n_rna = rna_dis_adj_mat.shape[0]
    n_dis = rna_dis_adj_mat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # k折交叉验证
    k_folds = 5
    final_auc = 0

    for _ in range(10):
        # 更新测试样本
        idx = load_data(rna_dis_adj_mat, k_folds)
        total_auc = 0

        for i in range(k_folds):
            print(f"********* {i + 1} of {k_folds}-flods *********")

            # 创建模型
            model = AGAEMD(
                n_rna + n_dis,
                args_config["num_hidden_layers"],
                args_config["num_embedding_features"],
                args_config["num_heads_per_layer"],
                n_rna,
                n_dis,
                device
            ).to(device)

            # 创建参数优化方案
            optimizer = Adam(model.parameters(), lr=args_config["lr"], weight_decay=args_config["weight_decay"])

            # 构建异构网络
            training_mat = rna_dis_adj_mat.copy()
            training_mat[tuple(idx[i])] = 0
            het_mat = construct_het_graph(training_mat, dis_sim_mat, rna_sim_mat, args_config["penalty_factor"])
            adj_mat = construct_adj_mat(training_mat)

            het_graph = torch.tensor(het_mat).to(device=device)
            adj_graph = torch.tensor(adj_mat).to(device=device)
            training_data = torch.tensor(training_mat).to(device=device)
            graph_data = (het_graph, adj_graph)

            n_pos_sample = np.sum(training_mat)
            pos_weight = torch.tensor((n_rna * n_dis - n_pos_sample) / n_pos_sample)
            norm = n_rna * n_dis / ((n_rna * n_dis - n_pos_sample) * 2)
            BEST_VAL_LOSS, PATIENCE_CNT = [inf, 0]

            for epoch in range(args_config["num_epoch"]):
                # train
                model.train()

                outputs = model(graph_data)
                truth_label = torch.reshape(training_data, (-1,))
                loss = calculate_loss(outputs, truth_label, norm, pos_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args_config["eval_freq"] > 0 and (epoch == 0 or (epoch + 1) % args_config["eval_freq"] == 0):
                    print(f"||{epoch + 1} of {args_config['num_epoch']}--------")
                    print(f"loss:{loss.item()}")

                if loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_LOSS = loss.item()
                    PATIENCE_CNT = 0
                else:
                    PATIENCE_CNT += 1
                    if PATIENCE_CNT > 100:
                        break

            # 计算auc
            model.eval()
            link_pred = model(graph_data).cpu().detach().numpy()
            auc = calculate_auc(rna_dis_adj_mat, link_pred, idx[i])
            print(f"AUC:{auc}")
            total_auc += auc / k_folds

        # 输出5折交叉验证均值结果
        print(f"{k_folds}-folds average auc:{total_auc}")
        final_auc += (total_auc / 10)

    print("____________________")
    print(f"|FINAL AUC:{round(final_auc * 100, 4)}%|")
    print("____________________")
    # 保存模型
    # torch.save(model, "AGAEMD.pth")
    return


if __name__ == '__main__':
    train_agaemd()
