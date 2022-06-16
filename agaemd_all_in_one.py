import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, normalize
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, JumpingKnowledge, GCNConv, GATConv, GATv2Conv, SAGEConv, Sequential
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from utils import load_data, calculate_rna_func_sim
from constants import Phase, INF


def calculate_loss(pred, pos_edge_idx, neg_edge_idx):
    pos_pred_socres = pred[pos_edge_idx[0], pos_edge_idx[1]]
    neg_pred_socres = pred[neg_edge_idx[0], neg_edge_idx[1]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0])))
    return binary_cross_entropy_with_logits(pred_scores, true_labels, reduction="mean")


# calculate AUC, AUPR, F1-score, accuracy
def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_labels = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))

    auc = roc_auc_score(true_labels, pred_labels)
    average_precision = average_precision_score(true_labels, pred_labels)

    pred_labels[np.where(pred_labels <= 0.5)] = 0
    pred_labels[np.where(pred_labels > 0.5)] = 1
    TP = np.dot(pred_labels, true_labels)
    FP = np.sum(pred_labels) - TP
    FN = np.sum(true_labels) - TP
    TN = len(true_labels) - TP - FP - FN

    accuracy = (TP + TN) / true_labels.shape[0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * TP / (true_labels.shape[0] + TP - TN)

    return np.array([average_precision, auc, f1_score, accuracy, precision, recall, specificity])


def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    rna_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    dis_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))

    mat1 = np.hstack((rna_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        # output = self.activation(output)
        output = output.mean(dim=1)

        return output

    def message(self, x_i, x_j, index, ptr, size_i):
        x = x_i + x_j
        x = self.leakyrelu(x)
        alpha = (x * self.attn_vec).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_dropout(alpha)

        return x_j * alpha.unsqueeze(-1)


class AGAEMD(nn.Module):
    def __init__(self, n_in_features: int, n_hid_layers: int, hid_features: list, n_heads: list, n_rna: int, n_dis: int,
                 add_layer_attn: bool, residual: bool, dropout: float = 0.6):
        super(AGAEMD, self).__init__()
        assert n_hid_layers == len(hid_features) == len(n_heads), f'Enter valid arch params.'
        self.n_rna = n_rna
        self.n_dis = n_dis
        self.n_hid_layers = n_hid_layers
        self.dropout = nn.Dropout(dropout)

        # stack graph attention layers
        self.conv = nn.ModuleList()
        tmp = [n_in_features] + hid_features
        for i in range(n_hid_layers):
            self.conv.append(
                GraphAttentionLayer(tmp[i], tmp[i + 1], n_heads[i], residual=residual),
                # GCNConv(tmp[i], tmp[i + 1], add_self_loops=False)
                # GATv2Conv(tmp[i], tmp[i + 1], n_heads[i], droposut=dropout, add_self_loops=True, concat=False)
                # GATConv(tmp[i], tmp[i + 1], n_heads[i], dropout=dropout, add_self_loops=True, concat=False)
            )

        if n_in_features != hid_features[0]:
            self.proj = Linear(n_in_features, hid_features[0], weight_initializer='glorot', bias=True)
        else:
            self.register_parameter('proj', None)

        if add_layer_attn:
            self.JK = JumpingKnowledge('lstm', tmp[-1], n_hid_layers + 1)
        else:
            self.register_parameter('JK', None)

        if self.proj is not None:
            self.proj.reset_parameters()

    def forward(self, x, edge_idx):
        # encoder
        embd_tmp = x
        embd_list = [self.proj(embd_tmp) if self.proj is not None else embd_tmp]
        for i in range(self.n_hid_layers):
            embd_tmp = self.conv[i](embd_tmp, edge_idx)
            embd_list.append(embd_tmp)

        if self.JK is not None:
            embd_tmp = self.JK(embd_list)

        final_embd = self.dropout(embd_tmp)

        # InnerProductDecoder
        rna_embd = final_embd[:self.n_rna, :]
        dis_embd = final_embd[self.n_rna:, :]
        ret = torch.mm(rna_embd, dis_embd.T)
        return ret


# phase: 0 - validation mode; 1 - test mode
def main(n_rna, n_dis, dis_semantic_sim, edge_idx_dict, args_config, device, phase):
    # initialize parameters
    lr = args_config['lr']
    weight_decay = args_config['weight_decay']
    kfolds = args_config['kfolds']
    residual = args_config['residual']
    add_layer_attn = args_config['add_layer_attn']
    num_epoch = args_config['num_epoch']
    num_hidden_layers = args_config['num_hidden_layers']
    num_heads_per_layer = [args_config['num_heads_per_layer'] for _ in range(num_hidden_layers)]
    num_embedding_features = [args_config['num_embedding_features'] for _ in range(num_hidden_layers)]

    model = AGAEMD(
        n_rna + n_dis, num_hidden_layers, num_embedding_features, num_heads_per_layer,
        n_rna, n_dis, add_layer_attn, residual
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = calculate_loss

    if phase == Phase.VALIDATION:
        # split miRNA-disease pair into multi-groups
        pos_edges = edge_idx_dict['training_edges']
        neg_edges = edge_idx_dict['training_neg_edges']

        idx = np.arange(pos_edges.shape[1])
        np.random.shuffle(idx)
        idx_splited = np.array_split(idx, kfolds)

        metrics_tensor = np.zeros((1, 7))
        for i in range(kfolds):
            print(f'################Fold {i + 1} of {kfolds}################')

            # create the labels of training set and validation set
            tmp = []
            for j in range(2, kfolds):
                tmp.append(idx_splited[(j + i) % kfolds])
            tmp = np.concatenate(tmp)

            training_message_edges = pos_edges[:, tmp]
            training_supervision_edges = pos_edges[:, idx_splited[i]]

            validation_message_edges = np.hstack((training_message_edges, training_supervision_edges))
            validation_edges = pos_edges[:, idx_splited[(i + 1) % kfolds]]

            # create corresponding negative samples for training and evaluting metrics
            i1 = training_supervision_edges.shape[1]
            i2 = validation_edges.shape[1]
            rng = np.random.default_rng()
            neg_edges_shuffled = rng.permutation(neg_edges, axis=1)
            training_neg_edges = neg_edges_shuffled[:, :i1]
            validation_neg_edges = neg_edges_shuffled[:, i1:i1 + i2]

            print('Dynamically calculate rna functional similarity..(train step)')
            rna_sim_mat = calculate_rna_func_sim(training_message_edges, dis_semantic_sim, n_rna, n_dis)
            print('Done!')

            # create a heterogenous and a adjacent matrixes
            rna_dis_adj_mat = np.zeros((n_rna, n_dis))
            rna_dis_adj_mat[training_message_edges[0], training_message_edges[1]] = 1
            het_mat = construct_het_mat(rna_dis_adj_mat, dis_semantic_sim, rna_sim_mat)
            adj_mat = construct_adj_mat(rna_dis_adj_mat)

            edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
            het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

            model.train()
            for epoch in range(num_epoch):
                pred_mat = model(het_mat_device, edge_idx_device).cpu()
                loss = loss_func(pred_mat, training_supervision_edges, training_neg_edges)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print('------EOPCH {} of {}------'.format(epoch + 1, args_config['num_epoch']))
                    print('Loss: {}'.format(loss))

            model.eval()
            with torch.no_grad():
                print('Dynamically calculate rna functional similarity..(validation step)')
                rna_sim_mat = calculate_rna_func_sim(validation_message_edges, dis_semantic_sim, n_rna, n_dis)
                print('Done!')

                rna_dis_adj_mat = np.zeros((n_rna, n_dis))
                rna_dis_adj_mat[validation_message_edges[0], validation_message_edges[1]] = 1
                het_mat = construct_het_mat(rna_dis_adj_mat, dis_semantic_sim, rna_sim_mat)
                adj_mat = construct_adj_mat(rna_dis_adj_mat)

                edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
                het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

                pred_mat = model(het_mat_device, edge_idx_device).cpu().detach().numpy()

                metrics = calculate_evaluation_metrics(pred_mat, validation_edges, validation_neg_edges)
                print(np.round(metrics * 100, 3))
                metrics_tensor += (metrics / kfolds)
        print('Average result: {}'.format(np.round(metrics_tensor * 100, 3)))

    elif phase == Phase.TEST:
        training_message_edges = edge_idx_dict['training_msg_edges'].copy()
        training_supervision_edges = edge_idx_dict['training_supervision_edges'].copy()
        training_neg_edges = edge_idx_dict['training_neg_edges'].copy()

        print('Dynamically calculate rna functional similarity..(train step)')
        rna_sim_mat = calculate_rna_func_sim(training_message_edges, dis_semantic_sim, n_rna, n_dis)
        print('Done!')

        # create a heterogenous and a adjacent matrixes
        rna_dis_adj_mat = np.zeros((n_rna, n_dis))
        rna_dis_adj_mat[training_message_edges[0], training_message_edges[1]] = 1
        het_mat = construct_het_mat(rna_dis_adj_mat, dis_semantic_sim, rna_sim_mat)
        adj_mat = construct_adj_mat(rna_dis_adj_mat)

        edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
        het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

        MINIMUM_LOSS, PATIENCE= [INF, 0]

        model.train()
        for epoch in range(num_epoch):
            pred_mat = model(het_mat_device, edge_idx_device).cpu()
            loss = loss_func(pred_mat, training_supervision_edges, training_neg_edges)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < MINIMUM_LOSS:
                MINIMUM_LOSS = loss.item()
                PATIENCE = 0
            else:
                PATIENCE += 1
                if PATIENCE > 100:
                    break

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print('------EOPCH {} of {}------'.format(epoch + 1, args_config['num_epoch']))
                print('Loss: {}'.format(loss))

        model.eval()
        with torch.no_grad():
            test_msg_edges = np.hstack((training_supervision_edges, training_message_edges))
            test_edges = edge_idx_dict['test_edges'].copy()
            test_neg_edges = edge_idx_dict['test_neg_edges'].copy()

            print('Dynamically calculate rna functional similarity..(test step)')
            rna_sim_mat = calculate_rna_func_sim(test_msg_edges, dis_semantic_sim, n_rna, n_dis)
            print('Done!')

            # create a heterogenous and a adjacent matrixes
            rna_dis_adj_mat = np.zeros((n_rna, n_dis))
            rna_dis_adj_mat[test_msg_edges[0], test_msg_edges[1]] = 1

            het_mat = construct_het_mat(rna_dis_adj_mat, dis_semantic_sim, rna_sim_mat)
            adj_mat = construct_adj_mat(rna_dis_adj_mat)

            edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
            het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

            model = torch.load('./AGAEMD_20220518.pth')
            pred_mat = model(het_mat_device, edge_idx_device).cpu().detach().numpy()

            metrics = calculate_evaluation_metrics(pred_mat, test_edges, test_neg_edges)
            print(np.round(metrics * 100, 3))
    torch.save(model, "./AGAEMD_20220518.pth")
    return metrics


if __name__ == '__main__':
    path = './Dataset_HMDD32'
    repeat_times = 1
    phase = Phase.TEST
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict = {
        'kfolds': 5,
        'num_heads_per_layer': 6,
        'num_embedding_features': 250,
        'num_hidden_layers': 4,
        'num_epoch': 5000,   # 5000
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'add_layer_attn': True,
        'residual': True,
    }

    final_result = np.zeros((repeat_times, 7))
    for i in range(repeat_times):
        print(f'********************{i + 1} of {repeat_times}********************')
        # load disease semantic similarity matrix and edge index of miRNA-disease adjacency matrix
        dis_semantic_sim, edge_idx_dict, n_rna, n_dis = load_data(path, phase, negative_sample_mode='new')

        reulst = main(n_rna, n_dis, dis_semantic_sim, edge_idx_dict, hyperparam_dict, device, phase)
        final_result[i, :] = reulst
    print("------------------------------------")
    print(np.mean(final_result, axis=0))
    print(np.std(final_result, axis=0))
