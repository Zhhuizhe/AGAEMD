import torch
import torch.nn as nn
import torch.nn.functional as F


# GAT
class GraphAttentionLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads,
                 in_drop=0.6, coef_drop=0.6, residual=True, concat=False, norm=False):
        super(GraphAttentionLayer, self).__init__()

        self.norm = norm
        self.concat = concat
        self.residual = residual

        # 创建投影矩阵(NH, F, F')
        self.W = nn.Parameter(torch.zeros((num_of_heads, num_in_features, num_out_features)))
        # 创建偏置矩阵(F', 1)
        self.bias = nn.Parameter(torch.zeros(num_of_heads, 1, num_out_features))
        # 该处与论文的实现有所区别
        self.scoring_fn_target = nn.Parameter(torch.zeros(num_of_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.zeros(num_of_heads, num_out_features, 1))

        # 初始化权值矩阵和偏置矩阵
        self.reset_parameters()

        # 初始化Dropout，激活函数
        self.coef_dropout = nn.Dropout(coef_drop)
        self.in_drop = nn.Dropout(in_drop)
        self.activation = nn.ELU()

        if residual:
            self.residual_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        if norm:
            self.instance_norm = nn.InstanceNorm2d(1, affine=False)

    def reset_parameters(self):
        # xavier初始化
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.zeros_(self.bias)

    def forward(self, data):
        in_nodes_features = data[0]
        connectivity_mask = data[1]
        num_of_nodes = in_nodes_features.shape[0]

        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        in_nodes_features = self.dropout(in_nodes_features)

        # 将图中结点的特征投影至相同特征空间
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.W)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # (NH, N+M, 1) + (NH, 1, N+M) -> (NH, N+M, N+M)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)
        logits = scores_source + torch.transpose(scores_target, 1, 2)
        attn_coefs = F.softmax(F.leaky_relu(logits, negative_slope=0.2) + connectivity_mask, dim=1)

        if self.in_drop != 0:
            in_nodes_features = self.in_drop(in_nodes_features)
        if self.coef_dropout != 0:
            attn_coefs = self.coef_dropout(attn_coefs)

        # (NH, N+M, N+M) * (NH, N+M, F') -> (NH, N+M, F')
        out_nodes_features = torch.bmm(attn_coefs, nodes_features_proj)
        out_nodes_features = out_nodes_features + self.bias

        # 残差结构
        if self.residual and self.norm:
            rows = out_nodes_features.shape[0]
            cols = out_nodes_features.shape[1]
            out_normalized = self.instance_norm(out_nodes_features.view((1, 1, rows, cols)))
            out_normalized = self.dropout(out_normalized.view(rows, cols))
            output_mat = out_normalized + self.residual_proj(in_nodes_features)
        elif self.residual and not self.norm:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                output_mat = out_nodes_features + in_nodes_features
            else:
                output_mat = out_nodes_features + self.residual_proj(in_nodes_features)

        # 选择以concat或average输出multi-heads结果
        if self.concat:
            output_mat = torch.cat(output_mat, dim=-1)
        else:
            output_mat = output_mat.mean(dim=0)

        output_mat = self.activation(output_mat)
        return (output_mat, connectivity_mask)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Decoder
class Decoder(nn.Module):
    def __init__(self, n_in_features, n_out_features, num_of_rna, num_of_dis):
        self.n_rna = num_of_rna
        self.n_dis = num_of_dis
        self.W_rna = nn.Parameter(torch.tensor(n_in_features, n_out_features))
        self.W_dis = nn.Parameter(torch.tensor(n_in_features, n_out_features))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W_rna)
        nn.init.xavier_uniform_(self.W_dis)

    def forward(self, inputs):
        embd_rna = inputs[:self.n_rna, :]
        embd_dis = inputs[self.n_rna:, :]

        proj_embd_rna = torch.mm(embd_rna, self.W_rna)
        proj_embd_dis = torch.mm(embd_dis, self.W_dis)

        ret = torch.mm(proj_embd_rna * proj_embd_dis)
        return torch.sigmoid(ret)


# HAN
class HGraphAttentionLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, num_of_rna, num_of_dis,
                 slope=0.2, dropout=0.6, concat=False, residual=True, norm=True, activation=nn.ELU(),):
        super(HGraphAttentionLayer, self).__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.num_of_rna = num_of_rna
        self.num_of_dis = num_of_dis

        self.concat = concat
        self.residual = residual
        self.norm = norm

        self.proj_rna = nn.Parameter(torch.zeros((num_of_heads, num_in_features, num_out_features)))
        self.proj_dis = nn.Parameter(torch.zeros((num_of_heads, num_in_features, num_out_features)))
        self.bias = nn.Parameter(torch.zeros(num_out_features))
        # 该处与论文的实现有所区别
        self.scoring_fn_target = nn.Parameter(torch.zeros(num_of_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.zeros(num_of_heads, num_out_features, 1))

        # 初始化权值矩阵和偏置矩阵
        self.reset_parameters()

        # 初始化LeakyReLU函数，Dropout，激活函数
        self.leakyrelu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.activation = activation

        if residual:
            self.residual_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        if norm:
            self.instance_norm = nn.InstanceNorm2d(1, affine=False)

    def reset_parameters(self):
        # xavier初始化
        nn.init.xavier_uniform_(self.proj_dis)
        nn.init.xavier_uniform_(self.proj_rna)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.zeros_(self.bias)

    def forward(self, data):
        input_mat = data[0]
        connectivity_mask = data[1]
        num_of_nodes = input_mat.shape[0]

        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        input_mat = self.dropout(input_mat)

        # 不同类型的结点使用不同的特征投影矩阵
        rna_features_proj = torch.matmul(input_mat[:, :self.num_of_rna].unsqueeze(0), self.proj_rna)
        dis_features_proj = torch.matmul(input_mat[:, self.num_of_rna:].unsqueeze(0), self.proj_dis)
        nodes_features_proj = torch.hstack((rna_features_proj, dis_features_proj))
        nodes_features_proj = self.dropout(nodes_features_proj)

        # (NH, N+M, 1) + (NH, 1, N+M) -> (NH, N+M, N+M)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)
        attn_coefs = self.leakyrelu(scores_source + torch.transpose(scores_target, 1, 2))
        attn_coefs = self.softmax(connectivity_mask + attn_coefs)
        # (NH, N+M, N+M) * (NH, N+M, F') -> (NH, N+M, F')
        vals = torch.bmm(attn_coefs, nodes_features_proj)

        # 选择以concat或average输出multi-heads结果
        if self.concat:
            # (NH, N+M, F') -> (N+M, NH, F')
            vals = vals.transpose(0, 1)
            vals = vals.view((-1, self.num_of_heads * self.num_out_features))
        else:
            vals = vals.mean(dim=0)



        # 残差结构
        if self.residual and self.norm:
            rows = vals.shape[0]
            cols = vals.shape[1]
            vals_norm = self.instance_norm(vals.view((1, 1, rows, cols)))
            vals_norm = self.dropout(vals_norm.view(rows, cols))
            output_mat = vals_norm + self.residual_proj(input_mat)
        elif self.residual and not self.norm:
            output_mat = vals + self.residual_proj(input_mat)

        output_mat = self.activation(output_mat)
        return (output_mat, connectivity_mask)


class GraphSAGELayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, residual=True, activation=nn.ELU()):

        self.activation = activation
        if residual:
            self.residual_proj = nn.Linear(num_in_features, num_out_features, bias=False)

    def sampler(self):
        pass

    def aggregate(self):
        pass

    def forward(self):
        pass


# GCN
class PlainLayer:
    def __init__(self, n_in_features, n_out_features, residual=True, dropout=0.6, activation=nn.ReLU()):
        self.residual = residual
        self.activation = activation

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.zeros((n_in_features, n_out_features)))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input_mat, adj_mat):
        output_mat = self.dropout(input_mat)
        output_mat = torch.matmul(output_mat, self.W)
        output_mat = self.activation(torch.mm(adj_mat, output_mat))
        # if self.residual


# MLP
class MLP(nn.Module):
    def __init__(self, dims, dropout=0.6, bias=True, activation=nn.ReLU()):
        num_in_features = dims[0]
        num_out_features = dims[-1]
        self.dims = dims
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.w_1 = nn.Parameter(torch.tensor((num_in_features, dims[1])))
        self.w_2 = nn.Parameter(torch.tensor((dims[1], num_out_features)))
        if bias:
            self.b_1 = nn.Parameter(torch.tensor(dims[1]))
            self.b_2 = nn.Parameter(torch.tensor(num_out_features))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.w_1)
        nn.init.xavier_uniform_(self.w_2)
        nn.init.xavier_uniform_(self.b_1)
        nn.init.xavier_uniform_(self.b_2)

    def forward(self, inputs):
        assert inputs.shape[-1] == self.w_1[0], \
            f'MLP: Wrong dimensions of input matrix(input=({inputs.shape[0]}, {inputs[1]})).'
        input = self.dropout(inputs)
        hid_output = self.activation(torch.matmul(input, self.w_1) + self.b_1)
        hid_output = self.dropout(hid_output)
        output = self.activation(torch.matmul(hid_output, self.w_2) + self.b_2)
        return output
