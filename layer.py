import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_head, coef_dropout=0.4, in_dropout=0.4, activation=nn.ELU(),
                 residual=True, concat=False, norm=False):
        super(GraphAttentionLayer, self).__init__()

        self.concat = concat
        self.residual = residual
        self.norm = norm

        # 创建投影矩阵(NH, F, F')，创建偏置向量(F', 1)
        self.W = nn.Parameter(torch.zeros((num_of_head, num_in_features, num_out_features)))
        self.bias = nn.Parameter(torch.zeros(num_out_features))
        # 该处与论文的实现有所区别
        self.scoring_fn_target = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.zeros(num_of_head, num_out_features, 1))

        # 初始化权值矩阵和偏置矩阵
        self.reset_parameters()

        # 初始化LeakyReLU函数，Dropout，激活函数
        self.in_dropout = nn.Dropout(in_dropout)
        self.coef_dropout = nn.Dropout(coef_dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
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
        num_of_features = in_nodes_features.shape[1]

        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        in_nodes_features = self.in_dropout(in_nodes_features)

        # 将图中结点的特征投影至相同特征空间
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.W)
        # nodes_features_proj = self.dropout(nodes_features_proj)

        # (NH, N+M, 1) + (NH, 1, N+M) -> (NH, N+M, N+M)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)
        attn_coefs = self.leakyrelu(scores_source + torch.transpose(scores_target, 1, 2))
        attn_coefs = self.softmax(connectivity_mask + attn_coefs)

        attn_coefs = self.coef_dropout(attn_coefs)
        nodes_features_proj = self.in_dropout(nodes_features_proj)

        # (NH, N+M, N+M) * (NH, N+M, F') -> (NH, N+M, F')
        vals = torch.bmm(attn_coefs, nodes_features_proj)

        # 残差结构
        if self.residual:
            if in_nodes_features.shape[-1] == vals.shape[-1]:
                vals += in_nodes_features
            else:
                vals += self.residual_proj(in_nodes_features)

        if self.activation:
            vals = self.activation(vals)

        # 输出multi-heads均值结果
        vals = vals.mean(dim=0)

        return (vals, connectivity_mask)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Decoder(nn.Module):
    def __init__(self, num_rna, num_dis, num_in_features, num_out_features=256, dropout=0.4):
        super(Decoder, self).__init__()
        self.num_rna = num_rna
        self.num_dis = num_dis
        self.dropout = nn.Dropout(dropout)
        self.W_rna = nn.Parameter(torch.zeros((num_in_features, num_out_features)))
        self.W_dis = nn.Parameter(torch.zeros((num_in_features, num_out_features)))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W_rna)
        nn.init.xavier_uniform_(self.W_dis)

    def forward(self, inputs):
        pass
