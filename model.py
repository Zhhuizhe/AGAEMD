import torch
import torch.nn as nn
from layer import GraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, n_in_features, n_hid_layers, n_embd_features, n_heads, n_mirna, n_disease, device, dropout=0.4, attn_coef_dropout=0.4, attn_in_dropout=0.4):
        super(AGAEMD, self).__init__()
        assert n_hid_layers == len(n_embd_features) == len(n_heads), f'Enter valid arch params.'

        self.num_rna = n_mirna
        self.num_dis = n_disease
        self.num_hid_layers = n_hid_layers
        self.num_in_features = n_in_features
        self.num_mid_features = n_embd_features[-1]
        self.device = device

        # 创建网络attention layer
        attn_layers = []
        for i in range(n_hid_layers):
            if i == 0:
                layer = GraphAttentionLayer(n_in_features, n_embd_features[i], n_heads[i], attn_coef_dropout, attn_in_dropout)
            else:
                layer = GraphAttentionLayer(n_embd_features[i - 1], n_embd_features[i], n_heads[i], attn_coef_dropout, attn_in_dropout)
            attn_layers.append(layer)
        self.net = nn.Sequential(
            *attn_layers,
        )

        # 创建权值矩阵，用于重建关联矩阵
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.zeros((n_embd_features[-1], n_embd_features[-1])))

        # 初始化
        nn.init.xavier_uniform_(self.weight)

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

        # BilinearDecoder
        # rna_embd = mid_out[:self.num_rna, :]
        # dis_embd = mid_out[self.num_rna:, :]
        # ret = torch.mm(rna_embd, self.weight)
        # ret = torch.sigmoid(ret)
        # ret = torch.mm(ret, torch.transpose(dis_embd, 0, 1))

        # InnerProductDecoder
        rna_embd = mid_out[:self.num_rna, :]
        dis_embd = mid_out[self.num_rna:, :]
        ret = torch.mm(rna_embd, dis_embd.T)
        ret = torch.sigmoid(ret)
        return torch.reshape(ret, (-1,))


