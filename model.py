import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import Decoder
from layer import GraphAttentionLayer
from layer import HGraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, n_hid_layers, n_embd_features, n_heads, n_mirna, n_disease, device,
                 drop=0.6, attn_ceof_drop=0.6, attn_in_drop=0.6, concat=True):
        super(AGAEMD, self).__init__()
        assert n_hid_layers == len(n_embd_features) - 1 == len(n_heads), f'Enter valid arch params.'

        self.num_rna = n_mirna
        self.num_dis = n_disease
        self.num_hid_layers = n_hid_layers
        self.num_mid_features = n_embd_features[-1]
        self.device = device

        # 创建网络GAT-based Encoder
        attn_layers = []
        n_heads = [1] + n_heads
        for i in range(n_hid_layers):
            if concat:
                layer = GraphAttentionLayer(n_embd_features[i] * n_heads[i], n_embd_features[i + 1], n_heads[i + 1], attn_in_drop, attn_ceof_drop, concat=concat)
            else:
                layer = GraphAttentionLayer(n_embd_features[i], n_embd_features[i + 1], n_heads[i + 1], attn_in_drop, attn_ceof_drop, concat=concat)
            attn_layers.append(layer)
        self.net = nn.Sequential(
            *attn_layers,
        )

        # 创建Decoder
        if concat:
            self.decoder = Decoder(n_embd_features[-1] * n_heads[-1], 256, n_mirna, n_disease)
        else:
            self.decoder = Decoder(n_embd_features[-1], 256, n_mirna, n_disease)

        # 创建权值矩阵，用于重建关联矩阵
        self.weight = nn.Parameter(torch.zeros((n_embd_features[-1], n_embd_features[-1])))
        self.dropout = nn.Dropout(drop)

    def init_parameters(self):
        # 初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, data):
        # encoder
        # mid_out_avg = torch.zeros((self.num_in_features, self.num_mid_features)).to(self.device)
        # mid_out = data
        # for i in range(self.num_hid_layers):
        #     mid_out = self.net[i](mid_out)
        #     mid_out_avg += mid_out[0]
        # mid_out_avg = mid_out_avg / self.num_hid_layers
        # mid_out = self.dropout(mid_out_avg)
        mid_out = self.net(data)[0]
        mid_out = self.dropout(mid_out)

        # decoder
        ret = self.decoder(mid_out)

        return torch.reshape(ret, (-1,))

    """ l2 regularization has been already add into the optimizer
    def loss(self, pred, label, norm, pos_weight):
        loss = norm * F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight, reduction="mean")
        return loss
    """


