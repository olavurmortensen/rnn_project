#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create word embeddings for the RNN using Word2Vec.
"""

from gensim.models import Word2Vec
import pickle

class SentenceGenerator(object):
    def __init__(self, item_list):
        self.item_list = item_list

    def __iter__(self):
        for item in self.item_list:
            yield item

if __name__ == '__main__':
    with open('data/OpenSubtitlesSentences.pickle', 'rb') as f:
        data = pickle.load(f)
    sentences = data['sentences']

    model = Word2Vec(SentenceGenerator(sentences), size=500, workers=8)

    vocab = {}
    for word, vocab_object in model.vocab.items():
        vocab[word] = vocab_object.index

    pickle.dump(model.syn0, open('../data/word_embeddings_vecs.pickle', 'wb'))
    pickle.dump(vocab, open('../data/word_embeddings_vocab.pickle', 'wb'))
