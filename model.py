import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, inputs, n_embd_features, attn_drop, slope, n_heads, n_mirna, n_disease):
        super(AGAEMD, self).__init__()

        self.n_rna = n_mirna
        self.n_dis = n_disease

        # 3层graph attention layer结构
        self.attn_layer1 = GraphAttentionLayer(inputs, n_embd_features[0], n_heads[0], attn_drop, slope)
        self.attn_layer2 = GraphAttentionLayer(n_embd_features[0], n_embd_features[1], n_heads[1], attn_drop, slope)
        self.attn_layer3 = GraphAttentionLayer(n_embd_features[1], n_embd_features[2], n_heads[2], attn_drop, slope)
        self.weight = nn.Parameter(torch.zeros((256, 256)))

        # xaiver初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, adj):
        # encoder
        mid_out = self.attn_layer1(inputs, adj)
        mid_out = self.attn_layer2(mid_out, adj)
        mid_out = self.attn_layer3(mid_out, adj)

        # decoder
        rna_embd = mid_out[:self.n_rna, :]
        dis_embd = mid_out[self.n_rna:, :]
        ret = torch.mm(rna_embd, self.weight)
        ret = torch.mm(ret, torch.transpose(dis_embd, 0, 1))
        return torch.reshape(ret, (-1,))


