import torch
import torch.nn as nn
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
        self.attn_ceof = nn.Parameter(torch.tensor([0.5, 0.33, 0.25]))

        # xaiver初始化
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(attn_drop)

    def forward(self, inputs, adj):
        # encoder
        embedding1 = self.attn_layer1(inputs, adj)
        embedding2 = self.attn_layer2(embedding1, adj)
        embedding3 = self.attn_layer3(embedding2, adj)
        mid_out = embedding1 * self.attn_ceof[0] + embedding2 * self.attn_ceof[1] + embedding3 * self.attn_ceof[2]
        mid_out = self.dropout(mid_out)
        # decoder
        rna_embd = mid_out[:self.n_rna, :]
        dis_embd = mid_out[self.n_rna:, :]
        ret = torch.mm(rna_embd, self.weight)
        ret = torch.mm(ret, torch.transpose(dis_embd, 0, 1))
        return torch.reshape(ret, (-1,))


