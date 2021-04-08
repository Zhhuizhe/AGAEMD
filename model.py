import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, inputs, hid_units, nb_classes, attn_drop, ffd_drop, slope, n_heads):
        super(AGAEMD, self).__init__()
        # 3层graph attention layer结构
        self.attns1 = [GraphAttentionLayer(inputs, hid_units[0], dropout=attn_drop, slope=slope, residual=True) for _ in range(n_heads[0])]
        self.attns2 = [GraphAttentionLayer(hid_units[0], hid_units[1], dropout=attn_drop, slope=slope, residual=True) for _ in range(n_heads[1])]
        self.attns3 = [GraphAttentionLayer(hid_units[1], nb_classes, dropout=attn_drop, slope=slope, residual=True) for _ in range(n_heads[2])]

    def forward(self, x, adj):
        # encoder
        mid_out = [attn(x, adj) for attn in self.attns1]
        mid_out = torch.sum(mid_out) / len(self.attns1)
        mid_out = [attn(mid_out, adj) for attn in self.attns1]
        mid_out = torch.sum(mid_out) / len(self.attns1)
        out = [attn(mid_out, adj) for attn in self.attns1]
        out = torch.sum(out) / len(self.attns1)

        # decoder
        rna_embd = out[:]
        dis_embd = out[:]
        ret = torch.mm()
        return ret


