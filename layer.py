import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_head, num_of_rna, num_of_dis,
                 dropout=0.6, slope=0.2, residual=True, concat=False, norm=True, activation=nn.ELU()):
        super(GraphAttentionLayer, self).__init__()

        self.norm = norm
        self.concat = concat
        self.residual = residual

        # 创建投影矩阵(NH, F, F')，创建偏置向量(F', 1)
        self.W = nn.Parameter(torch.zeros((num_of_head, num_in_features, num_out_features)))
        self.bias = nn.Parameter(torch.zeros(num_out_features))
        # 该处与论文的实现有所区别
        self.scoring_fn_target = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))

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
        attn_coefs = self.leakyrelu(scores_source + torch.transpose(scores_target, 1, 2))
        attn_coefs = self.softmax(connectivity_mask + attn_coefs)
        # (NH, N+M, N+M) * (NH, N+M, F') -> (NH, N+M, F')
        vals = torch.bmm(attn_coefs, nodes_features_proj)

        # 选择以concat或average输出multi-heads结果
        if self.concat:
            pass
        else:
            vals = vals.mean(dim=0)

        # 残差结构
        if self.residual and self.norm:
            rows = vals.shape[0]
            cols = vals.shape[1]
            vals_norm = self.instance_norm(vals.view((1, 1, rows, cols)))
            vals_norm = self.dropout(vals_norm.view(rows, cols))
            output_mat = vals_norm + self.residual_proj(in_nodes_features)
        elif self.residual and not self.norm:
            output_mat = vals + self.residual_proj(in_nodes_features)

        output_mat = self.activation(output_mat)
        return (output_mat, connectivity_mask)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 构建针对异构矩阵的layer结构
class HGraphAttentionLayer:
    def __init__(self, num_in_features, num_out_features, num_of_head, num_of_rna, num_of_dis, activation=nn.ELU(),
                 slope=0.2, dropout=0.6, concat=False, residual=True, norm=True):
        super(HGraphAttentionLayer, self).__init__()

        self.concat = concat
        self.num_of_rna = num_of_rna
        self.num_of_dis = num_of_dis
        self.residual = residual
        self.norm = norm

        self.proj_rna = nn.Parameter(torch.zeros((num_of_head, num_in_features, num_out_features)))
        self.proj_dis = nn.Parameter(torch.zeros((num_of_head, num_in_features, num_out_features)))
        self.bias = nn.Parameter(torch.zeros(num_out_features))
        # 该处与论文的实现有所区别
        self.scoring_fn_target = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))

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
        in_nodes_features = data[0]
        connectivity_mask = data[1]
        num_of_nodes = in_nodes_features.shape[0]

        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        in_nodes_features = self.dropout(in_nodes_features)

        # 将图中结点的特征投影至相同特征空间
        rna_features_proj = torch.matmul(in_nodes_features[:self.num_of_rna, :].unsqueeze(0), self.proj_rna)
        dis_features_proj = torch.matmul(in_nodes_features[self.num_of_rna:, :].unsqueeze(0), self.proj_dis)
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
            pass
        else:
            vals = vals.mean(dim=0)

        if self.norm:
            rows = vals.shape[0]
            cols = vals.shape[1]
            vals_norm = self.instance_norm(vals.view((1, 1, rows, cols)))
            vals_norm = vals_norm.view(rows, cols)

        # 加入残差结构
        if self.residual:
            vals_norm += self.residual_proj(in_nodes_features)

        output_mat = self.activation(vals_norm)

        return (output_mat, connectivity_mask)


class PlainLayer:
    def __init__(self, n_in_features, n_out_features, activation=nn.ReLU(), residual=True, dropout=0.6):
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


