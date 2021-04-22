import torch
import torch.nn as nn
from layer import GraphAttentionLayer
from layer import HGraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, n_in_features, n_hid_layers, n_embd_features, n_heads, drop, attn_drop, slope, n_mirna, n_disease, device):
        super(AGAEMD, self).__init__()
        assert n_hid_layers == len(n_embd_features) == len(n_heads), f'Enter valid arch params.'

        self.num_rna = n_mirna
        self.num_dis = n_disease
        self.num_hid_layers = n_hid_layers
        self.num_in_features = n_in_features
        self.num_mid_features = n_embd_features[-1]
        self.device = device

        # self.attn_ceof = nn.Parameter(torch.tensor([0.5, 0.33]))
        # 创建网络attention layer
        attn_layers = []
        for i in range(n_hid_layers):
            if i == 0:
                layer = GraphAttentionLayer(n_in_features, n_embd_features[i], n_heads[i], n_mirna, n_disease, attn_drop, slope)
            else:
                layer = GraphAttentionLayer(n_embd_features[i - 1], n_embd_features[i], n_heads[i], n_mirna, n_disease, attn_drop, slope)
            attn_layers.append(layer)
        self.net = nn.Sequential(
            *attn_layers,
        )
        # 创建权值矩阵，用于重建关联矩阵
        self.weight = nn.Parameter(torch.zeros((n_embd_features[-1], n_embd_features[-1])))

        # 初始化
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(drop)

    def forward(self, data):
        # encoder
        mid_out_avg = torch.zeros((self.num_in_features, self.num_mid_features)).to(self.device)
        mid_out = data
        for i in range(self.num_hid_layers):
            mid_out = self.net[i](mid_out)
            mid_out_avg += mid_out[0]
        mid_out_avg = mid_out_avg / self.num_hid_layers
        mid_out = self.dropout(mid_out_avg)
        # mid_out = self.net(data)[0]
        # mid_out = self.dropout(mid_out)

        # decoder
        rna_embd = mid_out[:self.num_rna, :]
        dis_embd = mid_out[self.num_rna:, :]
        ret = torch.mm(rna_embd, self.weight)
        ret = torch.mm(ret, torch.transpose(dis_embd, 0, 1))
        return torch.reshape(ret, (-1,))


