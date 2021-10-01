import torch
import torch.nn as nn
import torch.nn.functional as F

from layers2 import GraphAttentionLayer
from utils2 import log_sum_exp, argmax

class GAT2(nn.Module):

    def __init__(self, nfeat, nhid1, nhid2, nhid3, nhid4, embed, tag_to_ix, dropout, alpha):
        """Dense version of GAT."""
        super(GAT2, self).__init__()
        self.dropout = dropout
        self.embedding_dim = embed
        self.hidden_dim1 = nhid1
        self.hidden_dim2 = nhid2
        self.hidden_dim3 = nhid3
        self.hidden_dim4 = nhid4
        self.START = "<START>"
        self.STOP = "<STOP>"

        self.vocab_size = nfeat
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.embed1 = nn.Embedding(nfeat, embed)
        self.lstm1 = nn.LSTM(embed, nhid1//2, num_layers=1, bidirectional=True)  # Node Embedding

        self.gc1 = GraphAttentionLayer(nhid1, nhid2, dropout=dropout, alpha=alpha)  # Graph Embedding 1
        self.gc2 = GraphAttentionLayer(nhid2, nhid3, dropout=dropout, alpha=alpha)  # Graph Embedding 2

        # self.embed2 = nn.Embedding(nhid3, nhid3//2)
        self.lstm2 = nn.LSTM(nhid3, nhid4//2, num_layers=1, bidirectional=True)  # Graph Embedding + Token Embedding

        self.hidden2tag = nn.Linear(nhid4, self.tagset_size)      # Conditional Random Fields
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[self.START], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP]] = -10000

        # Initialize the parameters of LSTM
        self.hidden = self.init_hidden()
        self.hidden1 = self.init_hidden1()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim1 // 2),
                torch.randn(2, 1, self.hidden_dim1 // 2))

    def init_hidden1(self):
        return (torch.randn(2, 1, self.hidden_dim4 // 2),
                torch.randn(2, 1, self.hidden_dim4 // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[self.START]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP]]
        scores = log_sum_exp(terminal_var)
        return scores

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START]
        best_path.reverse()
        return path_score, best_path

    # Loss function loss
    def neg_log_likelihood(self, ids, sentence, tags, new_df):
        # print(sentence.shape)
        graph_embed = self._get_graph_embedding(ids, sentence, new_df)
        feats = self._get_lstm_features(ids, graph_embed, new_df)
        # print('nll feats : ', feats.shape)
        forward_score = self._forward_alg(feats)
        # print('nll forward_score : ', forward_score.shape)
        gold_score = self._score_sentence(feats, tags)
        # print('nll gold_score : ', gold_score.shape)
        return forward_score - gold_score

    def forward(self, ids, sentence, new_df):
        graph_embed = self._get_graph_embedding(ids, sentence, new_df)
        lstm_feats = self._get_lstm_features(ids, graph_embed, new_df)
        # print('lstm_out : ', lstm_feats.shape)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def _get_lstm_features(self, ids_int, sentence, new_df):
        # print('Graph LSTM Features Function Started .. ', sentence.shape)
        self.hidden1 = self.init_hidden1()
        # embeds = self.embed2(sentence).view(len(sentence), 1, -1)
        # print(embeds.shape)
        embeds1 = sentence.unsqueeze(dim=1)
        lstm_out, self.hidden1 = self.lstm2(embeds1, self.hidden1)
        # print(lstm_out.shape, self.hidden_dim4, len(sentence))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim4)
        # print(lstm_out.shape)
        lstm_feats = self.hidden2tag(lstm_out)
        # print('End of LSTM Features Function .. ')
        return lstm_feats

    def _get_graph_embedding(self, ids, sentence, new_df):
        # print('Graph Embedding Function Started .. ', sentence.shape, sentence)

        self.hidden = self.init_hidden()
        # print('hidden_shape : ', self.hidden[0].shape)

        embeds = self.embed1(sentence).view(len(sentence), 1, -1)
        # print(embeds.shape, embeds)
        lstm_out, self.hidden = self.lstm1(embeds, self.hidden)
        # print(lstm_out.shape)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim1)
        # print('lstm_out : ', lstm_out.shape)

        tij_rij = F.elu(self.gc1(ids, lstm_out, new_df, first=True))
        # print('graph_out 1 : ', tij_rij.shape)
        graph_out2 = F.elu(self.gc2(ids, tij_rij, new_df, first=False))
        # print('graph_out 2 : ', graph_out2.shape)

        # print('End of Graph Embedding Function .. ')

        return graph_out2


