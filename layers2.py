import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(self.in_features + 5, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, ids, lstm_out, edges_list, first):
        if first:

            lstm_dict = {}
            lstm_count = {}
            for i in range(lstm_out.shape[0]):
                if ids[i] not in lstm_dict:
                    lstm_dict[ids[i]] = np.array(lstm_out[i].tolist())
                    lstm_count[ids[i]] = 1
                else:
                    lstm_dict[ids[i]] = np.add(lstm_dict[ids[i]], lstm_out[i].tolist())
                    lstm_count[ids[i]] += 1

            final_lstm_dict = {}
            for key, value in lstm_dict.items():
                if key in lstm_count:
                    final_lstm_dict[key] = value / lstm_count[key]

            hij = torch.tensor(list(final_lstm_dict.values()), dtype=torch.float) 
            wh = torch.mm(hij, self.W)
            node_edge_embeddings = self._construct_node_edge_embeddings(wh, edges_list)

            zero_vec = -9e15 * torch.ones_like(node_edge_embeddings)
            a = np.ones((node_edge_embeddings.shape[0], node_edge_embeddings.shape[0]))
            np.fill_diagonal(a, 0)
            adj = torch.from_numpy(a)

            attention = torch.where(adj > 0, node_edge_embeddings, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            alpha = torch.mm(attention, wh)

            return alpha

        if not first:
            wh = self.leakyrelu(torch.mm(lstm_out, self.W))
            # wh = F.softmax(wh, dim=1)
            return wh

    def _construct_node_edge_embeddings(self, wh, rij):
        n = wh.size()[0]
        ti = wh.repeat_interleave(n, dim=0)
        tj = wh.repeat(n, 1)
        hij = torch.cat([ti, rij, tj], dim=1)
        node_edge_embeddings = self.leakyrelu(torch.matmul(hij, self.a).squeeze(1))
        return node_edge_embeddings.view(n, n)
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
