import numpy as np
import torch
import os
import enum
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid, normalize
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import negative_sampling
from scipy.sparse import coo_matrix
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score


INF = 99999


class Phase(enum.Enum):
    VALIDATION = 0
    TEST = 1

def calculate_rna_func_sim(edge_idx, dis_sim, n_rna, n_dis):
    rna_dis_mat = np.zeros((n_rna, n_dis))
    rna_dis_mat[edge_idx[0], edge_idx[1]] = 1

    num_of_rna = rna_dis_mat.shape[0]
    rna_func_sim = np.zeros((num_of_rna, num_of_rna))
    out_degree_vec = np.sum(rna_dis_mat, axis=1)
    out_degree_mat = out_degree_vec[:, None] + out_degree_vec

    dis_semantic_sim = dis_sim - np.diag(np.diag(dis_sim)) + np.eye(n_dis)

    related_diseases = [np.where(rna_dis_mat[i] == 1)[0] for i in range(num_of_rna)]
    for i in range(num_of_rna):
        DT_i = related_diseases[i]
        if len(DT_i) == 0:
            continue
        for j in range(i, num_of_rna):
            DT_j = related_diseases[j]
            if len(DT_j) == 0:
                continue
            rna_func_sim[i, j] = np.sum(np.max(dis_semantic_sim[DT_i, :][:, DT_j], axis=1)) + np.sum(np.max(dis_semantic_sim[DT_j, :][:, DT_i], axis=1))
            rna_func_sim[j, i] = rna_func_sim[i, j]
    rna_func_sim = np.divide(rna_func_sim, out_degree_mat, where=(out_degree_mat != 0))
    rna_func_sim = rna_func_sim - np.diag(np.diag(rna_func_sim))

    return rna_func_sim


# turn dense matrix into a sparse foramt
def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data


def load_data(path, phase, ratio: tuple = (0.2, 0.2), negative_sample_mode: str = 'normal'):
    # read disease semantic similarity matrix and adjacent matrix from existing files
    dis_semantic_sim = np.loadtxt(os.path.join(path, 'disease_semantic_similarity.txt'), dtype=float)
    mir_dis_adj = np.loadtxt(os.path.join(path, 'mir_dis_adj.txt'), dtype=int)

    n_rna, n_dis = mir_dis_adj.shape

    # remove self-loop in the disease semantic similarity matrix
    diag = np.diag(dis_semantic_sim)
    if np.sum(diag) != 0:
        dis_semantic_sim = dis_semantic_sim - np.diag(diag)

    # get the edge idx of positives samples
    rng = np.random.default_rng()
    pos_samples, edge_attr = dense2sparse(mir_dis_adj)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # get the edge index of negative samples
    if negative_sample_mode == 'normal':
        rng = np.random.default_rng()
        neg_samples = np.where(mir_dis_adj == 0)
        neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
    else:
        neg_samples = negative_sampling(torch.tensor(pos_samples, dtype=torch.long), num_nodes=(n_rna, n_dis)).numpy()
        neg_samples_shuffled = rng.permutation(neg_samples, axis=1)

    # split positive samples into training message samples, training supervision samples, test samples
    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]

    # r1 denotes the ratio between training supervision edges and whole edges
    # r2 denotes the ratio between testing edges and whole edges
    # In 'validation' state, the edge_idx_dict contains training edges, test edges and their corresponding negative samples. Test edges will not be used in this state.
    # In 'test' state, the edge_idx_dict contains training supervision edges, training message edges, test edges and their corresponding negative samples.
    r1, r2 = ratio
    idx_split = int(n_pos_samples * r2)
    edge_idx_dict['test_edges'] = pos_samples_shuffled[:, :idx_split]
    edge_idx_dict['test_neg_edges'] = neg_samples_shuffled[:, :idx_split]

    if phase == Phase.VALIDATION:
        edge_idx_dict['training_edges'] = pos_samples_shuffled[:, idx_split:]
        edge_idx_dict['training_neg_edges'] = neg_samples_shuffled[:, idx_split:]
    if phase == Phase.TEST:
        idx_split_tmp = idx_split + int(n_pos_samples * r1)

        edge_idx_dict['training_supervision_edges'] = pos_samples_shuffled[:, idx_split:idx_split_tmp]
        edge_idx_dict['training_neg_edges'] = neg_samples_shuffled[:, idx_split:idx_split_tmp]

        edge_idx_dict['training_msg_edges'] = pos_samples_shuffled[:, idx_split_tmp:]

    return dis_semantic_sim, edge_idx_dict, n_rna, n_dis


def calculate_loss(pred, pos_edge_idx, neg_edge_idx):
    pos_pred_socres = pred[pos_edge_idx[0], pos_edge_idx[1]]
    neg_pred_socres = pred[neg_edge_idx[0], neg_edge_idx[1]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0])))
    return binary_cross_entropy_with_logits(pred_scores, true_labels, reduction="mean")


# calculate AUC, AUPR, F1-score, accuracy
# The metrics calculations is derived from the TDRC
def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_scores = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))

    auc = roc_auc_score(true_labels, pred_scores)
    average_precision = average_precision_score(true_labels, pred_scores)

    pred_scores_mat = np.mat([pred_scores])
    true_labels_mat = np.mat([true_labels])
    sorted_predict_score = np.array(sorted(list(set(np.array(pred_scores_mat).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(pred_scores_mat, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix * true_labels_mat.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = true_labels_mat.sum() - TP
    TN = len(true_labels_mat.T) - TP - FP - FN
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(true_labels_mat.T) + TP - TN)
    accuracy_list = (TP + TN) / len(true_labels_mat.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]

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

        output = self.activation(output)
        output = output.mean(dim=1)
        # output = normalize(output, p=2., dim=-1)

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
            )

        if n_in_features != hid_features[0]:
            self.proj = Linear(n_in_features, hid_features[0], weight_initializer='glorot', bias=True)
        else:
            self.register_parameter('proj', None)

        if add_layer_attn:
            self.JK = JumpingKnowledge('cat', tmp[-1], n_hid_layers + 1)
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

            pred_mat = model(het_mat_device, edge_idx_device).cpu().detach().numpy()

            metrics = calculate_evaluation_metrics(pred_mat, test_edges, test_neg_edges)
            print(np.round(metrics * 100, 3))
            torch.save(model, 'AGAEMD_hmddv3_RES.pt')

    return metrics


if __name__ == '__main__':
    path = '../Dataset_HMDD32'
    repeat_times = 1
    phase = Phase.TEST
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict = {
        'kfolds': 5,
        'num_heads_per_layer': 6,
        'num_embedding_features': 200,
        'num_hidden_layers': 4,
        'num_epoch': 5000,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'add_layer_attn': True,
        'residual': False,
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
