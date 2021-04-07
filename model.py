import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class AGAEMD(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, slope, nhead):
        super(AGAEMD, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, slope=slope) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.weight1 = nn.Parameter(torch.zeros(size=(nfeat, nhid)))
        self.weight2 = nn.Parameter(torch.zeros(size=(nfeat, nhid)))
        self.weight3 = nn.Parameter(torch.zeros(size=(nfeat, nhid)))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


