import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # print('initial shapes : ', h.shape, self.W.shape, self.a.shape, self.in_features, self.out_features)
        wh = torch.mm(h, self.W)
        # print('wh: ', wh.shape)
        hij = self._concat_features(wh)
        # print('hij: ', hij.shape)
        alpha = self.leakyrelu(torch.matmul(hij, self.a).squeeze(2))
        # print('alpha : ', alpha.shape)
        alpha = torch.mm(alpha, adj)
        # print('alpha1 : ', alpha.shape)
        alphaij = F.softmax(alpha, dim=1)
        # print('alphaij : ', alphaij.shape)
        # alphaij = F.dropout(alphaij, self.dropout, training=self.training)
        ti = torch.matmul(alphaij, wh)
        # print('ti : ', ti.shape)

        # if self.concat:
        #     return F.elu(ti)
        # else:
        return ti

    def _concat_features(self, h):
        N = h.size()[0]
        # print(N)
        h_repeated_in_chunks = h.repeat_interleave(N, dim=0)
        # print('h_repeated_in_chunks : ', h_repeated_in_chunks.shape)
        h_repeated_alternating = h.repeat(N, 1)
        # print('h_repeated_alternating : ', h_repeated_alternating.shape)
        all_combinations_matrix = torch.cat([h_repeated_in_chunks, h_repeated_alternating], dim=1)
        # print('all_combinations_matrix : ', all_combinations_matrix.shape)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'