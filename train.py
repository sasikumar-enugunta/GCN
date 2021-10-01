from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from model2 import GAT2
from utils2 import load_data, prepare_sequence, compare, load_edge_embed_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

parser.add_argument('--embedding_dim', type=int, default=500, help='Number of hidden units in LSTM layer.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--hidden4', type=int, default=32, help='Number of hidden units.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

idx_features_labels, word_to_ix, tag_to_ix = load_data()
edge_embeddings = load_edge_embed_data()
# print(len(idx_features_labels), len(word_to_ix), len(tag_to_ix))

# Model and optimizer
model = GAT2(
            nfeat=len(word_to_ix),
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nhid3=args.hidden3,
            nhid4=args.hidden4,
            embed=args.embedding_dim,
            tag_to_ix=tag_to_ix,
            dropout=args.dropout,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

training_data, test_data = train_test_split(idx_features_labels, test_size=0.2, random_state=0)
print('No. Documents : ', len(idx_features_labels), '\nNo. Words : ', len(word_to_ix), '\nNo. Tags : ', len(tag_to_ix),
      '\nTrain Size : ', len(training_data), '\nTest Size : ', len(test_data), '\n==========================')


def train(epochs):
    for epoch in range(epochs):
        print('Epoch: {:04d}'.format(epoch+1))
        t = time.time()
        count = 1

        for ids, sentence, tags, bold, underline, color in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)

            new_tags = {}
            for i in range(len(ids)):
                if ids[i] not in new_tags:
                    new_tags[ids[i]] = tags[i]

            new_tag_list = list(new_tags.values())

            targets = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long)

            ids_int = [int(i) for i in ids]
            df1 = edge_embeddings.loc[edge_embeddings['src'].isin(ids_int)]
            new_df = df1[['src', 'hori_dist', 'vert_dist', 'ar_one', 'ar_two', 'ar_three', 'dest']]

            loss = model.neg_log_likelihood(ids_int, sentence_in, targets, new_df)

            loss.backward()
            optimizer.step()

            if count % 100 == 0:
                print("Iteration %d : loss %f " % (count, loss))
            count += 1
            
        print('time: {:.4f}s'.format(time.time() - t))


def compute_test():
    sumloss = 0
    with torch.no_grad():
        for pair in test_data[0:10]:
            ids = pair[0]
            sentence = pair[1]
            tag = pair[2]

            new_tags = {}
            for i in range(len(ids)):
                if ids[i] not in new_tags:
                    new_tags[ids[i]] = tag[i]

            new_tag_list = list(new_tags.values())

            ids_int = [int(i) for i in ids]
            df1 = edge_embeddings.loc[edge_embeddings['src'].isin(ids_int)]
            new_df = df1[['src', 'hori_dist', 'vert_dist', 'ar_one', 'ar_two', 'ar_three', 'dest']]

            precheck_sent = prepare_sequence(sentence, word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in new_tag_list], dtype=torch.long)
            score, y_pred = model(ids_int, precheck_sent, new_df)

            print("Actual : ", list(precheck_tags.numpy()))
            print("Predct : ", y_pred)
            errorindex = compare(precheck_tags, y_pred)
            sumloss = sumloss + (len(errorindex) / len(precheck_tags))
            accuracy = (len(precheck_tags) - len(errorindex)) / len(precheck_tags)
            print('Loss = {:.4f}'.format((len(errorindex) / len(precheck_tags))))
            print("Error_indices : ", errorindex)
            print('Accuracy = {:.4f}'.format(accuracy))
            print("====================")

        print('Total Loss : ', sumloss / len(test_data))


epochs = 3
t_total = time.time()
train(epochs)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

compute_test()
