# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import numpy as np
import pyhanlp
from gensim.models import KeyedVectors


mode = 0
char_size = 128
maxlen = 512

word2vec = KeyedVectors.load_word2vec_format('data/financial.word.txt')

id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.wv.vectors
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1,word_size)),word2vec])

# def tokenize(s):
#     return [i for i in jieba.lcut(s)]

def tokenize2(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def seq_padding(X,padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x,[padding]*(ML-len(x))]) if len(x) < ML else x for x in X
        ])

def sent2vec(S):
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w,0))
    V = seq_padding(V)
    V = word2vec[V]
    return V

print (tokenize2("今天天气很好"))