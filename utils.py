import numpy as np
import scipy.sparse as sp
import torch


def load_data(path="./data/invoice/", dataset="invoice"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.new_feature_file".format(path, dataset), dtype=np.int32)
    features = np.array(idx_features_labels[:, 1:10], dtype=np.float32)
    labels = np.array(idx_features_labels[:, -7:], dtype=np.int32)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    edges_unordered = np.genfromtxt("{}{}.edges".format(path, dataset), dtype=np.int32)
    edges = np.array(edges_unordered[:, :2], dtype=np.int32)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(1400)         # 26 receipts for training
    idx_val = range(1400, 1548)     # 5 receipts for validation
    idx_test = range(1548, 1756)    # 5 receipts for testing

    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(np.array(adj.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels, dummy):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if dummy == 1:
        print(preds, correct)
    return correct / len(labels)

