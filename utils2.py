import torch
import pandas as pd
import numpy as np


def compare(y, y_pred):
    error_index = []
    if len(y) == len(y_pred):
        for i in range(0, len(y)):
            if y[i] != y_pred[i]:
                error_index.append(i)

    return error_index


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]    # to_ix.has_key(x)
    return torch.tensor(idxs, dtype=torch.long)


def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def prepare_pad(sentences, ids):
    # print(ids, sentences)
    max_seq_len = get_maximum_sequence_length(ids)
    # print(max_seq_len)
    main_list = []
    temp_list = []
    for i in range(len(ids)-1):
        if ids[i] != ids[i+1]:
            temp_list.append(sentences[i])
            if temp_list:
                main_list.append(temp_list)
                temp_list = []
        elif ids[i] == ids[i+1]:
            temp_list.append(sentences[i])
            if i+1 == len(ids)-1 and temp_list:
                temp_list.append(sentences[i+1])
                main_list.append(temp_list)
                temp_list = []
    # print(main_list)
    features = pad_input(main_list, max_seq_len)
    # print(features)
    return features


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_maximum_sequence_length(ids):
    count = 0
    for id_ in ids:
        curr_frequency = ids.count(id_)
        if curr_frequency > count:
            count = curr_frequency
    return count


def readFile1(ner_file):

    data1 = []

    ids = []
    sentences = []
    labels = []
    bold = []
    underline = []
    color = []

    seq_id = []
    seq_data = []
    seq_label = []
    seq_bold = []
    seq_underline = []
    seq_color = []
    for i in ner_file.readlines():
        i = i.replace("\n", "")
        lst = i.split(",")
        if len(lst) == 6:
            if len(lst[1]) == 1:
                seq_id.append(lst[0])
                seq_data.append(lst[1])
                seq_label.append(lst[2])
                seq_bold.append(lst[3])
                seq_underline.append(lst[4])
                seq_color.append(lst[5])
            else:
                seq_data.append(lst[1])
                for k in range(len(lst[1].split(' '))):
                    seq_id.append(lst[0])
                    seq_label.append(lst[2])
                    seq_bold.append(lst[3])
                    seq_underline.append(lst[4])
                    seq_color.append(lst[5])

        else:
            idx = " ".join(seq_id)
            seq_id.clear()
            ids.append(idx)

            sent = " ".join(seq_data)
            seq_data.clear()
            sentences.append(sent)

            label = " ".join(seq_label)
            seq_label.clear()
            labels.append(label)

            bold_ = " ".join(seq_bold)
            seq_bold.clear()
            bold.append(bold_)

            underline_ = " ".join(seq_underline)
            seq_underline.clear()
            underline.append(underline_)

            color_ = " ".join(seq_color)
            seq_color.clear()
            color.append(color_)

    for i in range(len(sentences)):
        data1.append((ids[i].split(), sentences[i].split(), labels[i].split(), bold[i].split(), underline[i].split(), color[i].split()))

    return data1


def getVocab(idx_features_labels):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {}
    word_to_ix = {"<UNK>": 0}  # Vocabulary to id=0

    for ids, sentence, tags, bold, underline, color in idx_features_labels:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

        for bold_ in bold:
            if bold_ not in word_to_ix:
                word_to_ix[bold_] = len(word_to_ix)

        for underline_ in underline:
            if underline_ not in word_to_ix:
                word_to_ix[underline_] = len(word_to_ix)

        for color_ in color:
            if color_ not in word_to_ix:
                word_to_ix[color_] = len(word_to_ix)

        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    tag_to_ix[START_TAG] = 7
    tag_to_ix[STOP_TAG] = 8

    return word_to_ix, tag_to_ix


def load_data(path="./data/invoice/", dataset="invoice"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    ner_file = open("{}{}.final_feature_embeddings".format(path, dataset), encoding="utf-8")
    idx_features_labels = readFile1(ner_file)
    word_to_ix1, tag_to_ix = getVocab(idx_features_labels)
    return idx_features_labels, word_to_ix, tag_to_ix


def load_edge_embed_data(path="./data/invoice/", dataset="invoice"):
    edge_file = open("{}{}.final_edge_embeddings".format(path, dataset), encoding="utf-8")
    df = pd.read_csv(edge_file)
    return df
