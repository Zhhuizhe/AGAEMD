import torch
import torch.nn as nn
from layer import GraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, n_in_features, n_hid_layers, n_embd_features, n_heads, attn_drop, slope, n_mirna, n_disease):
        super(AGAEMD, self).__init__()
        assert n_hid_layers + 1 == len(n_embd_features) == len(n_heads), f'Enter valid arch params.'

        self.n_rna = n_mirna
        self.n_dis = n_disease

        # self.attn_ceof = nn.Parameter(torch.tensor([0.5, 0.33, 0.25]))
        # 创建网络attention layer
        attn_layers = []
        for i in range(n_hid_layers):
            if i == 0:
                layer = GraphAttentionLayer(n_in_features, n_embd_features[i], n_heads[i], attn_drop, slope)
            else:
                layer = GraphAttentionLayer(n_embd_features[i], n_embd_features[i + 1], n_heads[i], attn_drop, slope)
            attn_layers.append(layer)
        self.net = nn.Sequential(
            *attn_layers,
        )
        # 创建权值矩阵，用于重建关联矩阵
        self.weight = nn.Parameter(torch.zeros((n_embd_features[-1], n_embd_features[-1])))

        # 初始化
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(attn_drop)

    def forward(self, data):
        # encoder
        mid_out = self.net(data)[0]
        mid_out = self.dropout(mid_out)

        # decoder
        rna_embd = mid_out[:self.n_rna, :]
        dis_embd = mid_out[self.n_rna:, :]
        ret = torch.mm(rna_embd, self.weight)
        ret = torch.mm(ret, torch.transpose(dis_embd, 0, 1))
        return torch.reshape(ret, (-1,))


