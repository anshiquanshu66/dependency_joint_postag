__author__ = 'Dung Doan'

from gensim.models.word2vec import Word2Vec
import numpy as np
import torch.nn as nn


def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    print('load embedding: %s from %s' % (embedding, embedding_path))
    if embedding == 'word2vec':
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        embedd_dim = -1
        embedd_dict = dict()
        count = 0
        with open(embedding_path, 'rb') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    if (embedd_dim + 1 != len(tokens)):
                        continue
                    # assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]

                word = tokens[0].decode('utf-8')
                word = word.replace("_", " ")

                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

def load_embedding_dict_bert(bert_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(bert_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split()

            if len(tokens) ==2:
                continue

            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if (embedd_dim + 1 != len(tokens)):
                    continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = tokens[1:]

            word = tokens[0]
            embedd_dict[word] = embedd
    return embedd_dict, embedd_dim

def freeze_embedding(embedding):
    assert isinstance(embedding, nn.Embedding), "input should be an Embedding module."
    embedding.weight.detach_()