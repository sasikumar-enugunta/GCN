import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nhid4, embed, nclass, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.embed1 = nn.Embedding(nfeat, embed)
        self.lstm1 = nn.LSTM(embed, nhid1, num_layers=1, bidirectional=True, batch_first=True)   # node embedding
        self.fc1 = nn.Linear(nhid1 * 2, nhid1)

        self.gc1 = GraphAttentionLayer(nhid1, nhid2, dropout=dropout, alpha=alpha, concat=True)
        self.gc2 = GraphAttentionLayer(nhid2, nhid3, dropout=dropout, alpha=alpha, concat=False)

        # GRAPH EMBEDDINGS (GRAPH CONVOLUTION LAYER) +  TOKEN EMBEDDINGS (WORD2VEC) -> PASS TO LSTM LAYER
        self.embed2 = nn.Embedding(nhid3,  nhid4)
        self.lstm2 = nn.LSTM(nhid4, nhid4, num_layers=1, bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(nhid4*2, nclass)

    def forward(self, x, adj):
        x = self.embed1(x.long())
        # print(x.shape)
        x, _ = self.lstm1(x)
        # print('lstm1 : ', x.shape)
        x = x[:, -1, :]
        # print(x.shape)
        x = self.fc1(x)
        # print('fc1 : ', x.shape)

        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc1(x, adj))
        # print('attentions : ', x.shape)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gc2(x, adj))
        # print('elu : ', x.shape)

        x = self.embed2(x.long())
        # print(x.shape)
        x, _ = self.lstm2(x)
        # print('lstm2 : ', x.shape)
        x = x[:, -1, :]
        # print(x.shape)
        x = self.fc2(x)
        # print('fc2 : ', x.shape)

        return F.log_softmax(x, dim=1)

