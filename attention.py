import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class HyperedgeAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(HyperedgeAttention, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, X, H):
        N = X.shape[0]
        X = X.reshape(N, -1)
        X = torch.mm(H.T, X)
        w = self.layers(X)
        beta = torch.softmax(w, dim=0)
        return beta * X


class AttentionLayer(nn.Module):
    def __init__(self, in_size, out_size, layer_num_heads, dropout):
        super(AttentionLayer, self).__init__()

        self.gat_layer = GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu)
        # self.hgnn = HGNN_embedding(in_size, out_size)
        self.hyperedge_attention = HyperedgeAttention(in_size=out_size * layer_num_heads)

    def forward(self, g, h, H):  # G
        hyperedge_embeddings = self.gat_layer(g, h)  # self.hgnn(h, G)
        return self.hyperedge_attention(hyperedge_embeddings, H)


class Multi_view_Attention(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_heads, dropout):
        super(Multi_view_Attention, self).__init__()
        self.dropout = dropout
        self.layer = nn.ModuleList()
        self.layer.append(AttentionLayer(in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layer.append(AttentionLayer(hidden_size * num_heads[l-1], hidden_size, num_heads[l], dropout))
        self.fc = nn.Linear(hidden_size*num_heads[-1], out_size)

    def forward(self, g, h, H):  # G
        for layer in self.layer:
            h = layer(g, h, H)  # G
            F.dropout(h, self.dropout)

        return self.fc(h)


