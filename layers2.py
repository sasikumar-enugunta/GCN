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

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features // 2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc1 = nn.Linear(2 * self.out_features + 5, self.out_features // 2)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, ids, lstm_out, new_df, first):
        # print('initial shapes : ', lstm_out.shape, self.W.shape, self.a.shape, self.in_features, self.out_features)

        if first:

            lstm_dict = {}
            for i in range(len(ids)):
                if ids[i] not in lstm_dict:
                    lstm_dict[ids[i]] = np.array(lstm_out[i].tolist())
                else:
                    lstm_dict[ids[i]] = np.add(lstm_dict[ids[i]], lstm_out[i].tolist()) / 2

            h = torch.tensor(list(lstm_dict.values()), dtype=torch.float)       # node embeddings

            wh = torch.mm(h, self.W)
            # print('wh : ', wh, wh.shape)

            node_edge_embeddings = self._construct_node_edge_embeddings(wh, new_df)	# construct node_edge_embeddings
            # print('node_edge_embeddings : ', node_edge_embeddings.shape)

            a3 = nn.Parameter(torch.empty(size=(node_edge_embeddings.shape[0], node_edge_embeddings.shape[1])))
            nn.init.xavier_uniform_(a3.data, gain=1.414)

            e = self.leakyrelu(torch.matmul(a3.T, node_edge_embeddings))	# multiplying node_edge_embeddings to get the shape of N*N
            # print('e : ', e, e.shape)

            zero_vec = -9e15 * torch.ones_like(e)

            a = np.ones((zero_vec.shape[0], zero_vec.shape[0]))
            np.fill_diagonal(a, 0)
            adj = torch.from_numpy(a)			# creating adjacency matrix for self-attention mechanism

            # adj1 = torch.where(e > 0, 1, 0)

            attention = torch.where(adj > 0, e, zero_vec)
            # print('attention : ', attention, attention.shape)

            attention = F.softmax(attention, dim=1)
            # print('attention : ', attention, attention.shape)
            h_prime = torch.matmul(attention, wh)
            # print('h_prime : ', h_prime, h_prime.shape)
            return h_prime

        if not first:
            wh = self.leakyrelu(torch.mm(lstm_out, self.W))
            # wh = F.softmax(wh, dim=1)
            return wh

    def _construct_node_edge_embeddings(self, wh, new_df):
        n = wh.size()[0]
        wh_repeated_in_chunks = wh.repeat_interleave(n, dim=0)
        wh_repeated_alternating = wh.repeat(n, 1)

        # print(wh_repeated_in_chunks.shape, wh_repeated_alternating.shape)

        # numpy_df = new_df.to_numpy()
        # numpy_edges = numpy_df[:, 1:6]
        # final_edges = torch.FloatTensor(numpy_edges)

        final_list = []
        for src_idx, src_row in new_df.iterrows():
            temp_list1 = [float(src_row['hori_dist']), float(src_row['vert_dist']), float(src_row['ar_one']),
                          float(src_row['ar_two']), float(src_row['ar_three'])]
            final_list.append(temp_list1)
        final_list = torch.FloatTensor(final_list)

        all_combinations_matrix = torch.cat([wh_repeated_in_chunks, wh_repeated_alternating, final_list], dim=1)

        a1 = nn.Parameter(torch.empty(size=(self.in_features + 5, n)))
        nn.init.xavier_uniform_(a1.data, gain=1.414)
        node_edge_embeddings = torch.mm(all_combinations_matrix, a1)

        node_edge_embeddings = node_edge_embeddings.view(n * n, n)

        return node_edge_embeddings

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
