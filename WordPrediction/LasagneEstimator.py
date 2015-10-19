#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Sequence learning using LasagneNet.
'''

from __future__ import absolute_import

from data_generator import load_sentences
from LasagneNet import *
from RepeatLayer import RepeatLayer
import os
from time import time
import lasagne
import theano.tensor as T
import numpy as np
import theano
import cPickle as pickle

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def encode_str(text, token_dict, max_seq_len):
    text = ([c for c in text if c in token_dict])
    text = text[0:max_seq_len]
    enc_str = [token_dict[c] for c in text]
    remainder = max_seq_len-len(text)
    enc_str = np.pad(enc_str, ((0, remainder)), mode='constant', constant_values=(((0, 0))))
    mask = np.asarray([1]*len(text) + [0]*remainder, dtype=np.float32)
    return enc_str, mask

def word_prediction_network(BATCH_SIZE, EMBEDDING_SIZE, NUM_WORDS, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU, learning_rate):
    # Create data for testing network dimensions
    x_sym = T.imatrix()
    y_sym = T.imatrix()
    xmask_sym = T.matrix()

    NUM_OUTPUTS = int(NUM_WORDS + 1)
    
    X = np.random.randint(0, NUM_WORDS, size=(BATCH_SIZE, MAX_SEQ_LEN)).astype('int32')
    Xmask = np.ones((BATCH_SIZE, MAX_SEQ_LEN)).astype('float32')

    l_in = lasagne.layers.InputLayer((None, MAX_SEQ_LEN), name='input')  # TODO: BATCH_SIZE instead of "None"?
    print "l_in shape: %s" % str((lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: X}).shape))

    l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_WORDS, EMBEDDING_SIZE,
                                              W=WEIGHTS.astype('float32'),
                                              name='embedding')

    l_emb.params[l_emb.W].remove('trainable')
    print "l_emb shape: %s" % str((lasagne.layers.get_output(l_emb, inputs={l_in: x_sym}).eval({x_sym: X}).shape))

    l_mask = lasagne.layers.InputLayer((None, MAX_SEQ_LEN), name='input_mask')

    l_rec_for = lasagne.layers.LSTMLayer(l_emb, num_units=NUM_UNITS_GRU, name='rec_for', mask_input=l_mask)
    print "rec_for shape: %s" % str(lasagne.layers.get_output(l_rec_for, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)
   
    # slice last index of dimension 1
    l_last_hid_for = lasagne.layers.SliceLayer(l_rec_for, indices=-1, axis=1, name='last_hid_for')
    print  "last_hid_for shape: %s" % str(lasagne.layers.get_output(l_last_hid_for, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)
    
    l_rec_bac = lasagne.layers.LSTMLayer(l_emb, backwards=True, num_units=NUM_UNITS_GRU, name='rec_bac', mask_input=l_mask)
    print "rec_bac shape: %s" % str(lasagne.layers.get_output(l_rec_bac, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)
    
    # slice last index of dimension 1
    l_last_hid_bac = lasagne.layers.SliceLayer(l_rec_bac, indices=-1, axis=1, name='last_hid_bac')
    print  "last_hid_bac shape: %s" % str(lasagne.layers.get_output(l_last_hid_bac, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)

    l_concat = lasagne.layers.ConcatLayer(incomings=[l_last_hid_for, l_last_hid_bac], name='concat')

    l_softmax = lasagne.layers.DenseLayer(l_concat, num_units=NUM_OUTPUTS,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          name='softmax')
    print "l_softmax = DenseLayer: %s" % str(lasagne.layers.get_output(l_softmax, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)

    output_train = lasagne.layers.get_output(l_softmax,
            inputs={l_in: x_sym, l_mask: xmask_sym},
            deterministic=False)

    total_cost = T.nnet.categorical_crossentropy(
        T.reshape(output_train, (-1, NUM_OUTPUTS)), y_sym.flatten())
    mean_cost = T.mean(total_cost) #cost expression

    argmax = T.argmax(output_train, axis=-1)
    eq = T.eq(argmax, y_sym.flatten())
    acc = T.mean(eq)  #accuracy

    all_trainable_parameters = lasagne.layers.get_all_params([l_softmax], trainable=True)

    #add grad clipping to avoid exploding gradients
    all_grads = [T.clip(g, -3, 3) for g in T.grad(mean_cost, all_trainable_parameters)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads, 3)

    updates = lasagne.updates.adam(
            all_grads,
            all_trainable_parameters,
            learning_rate=learning_rate,) # adaptive learning rate should be implemented...

    train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc], updates=updates)
    test_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc])
    predict_func = theano.function([x_sym, xmask_sym], [output_train])

    # when the input X is a dict, the following definitions will allow LasagneNet to call train_func without
    # knowing the order of the inputs, using the syntax train_function(**X)
    def train_function(X, y, X_mask):
        return train_func(X, y, X_mask)

    def test_function(X, y, X_mask):
        return test_func(X, y, X_mask)

    def predict_function(X, X_mask):
        return predict_func(X, X_mask)

    return l_softmax, train_function, test_function, predict_function


if __name__ == "__main__":
    learning_rate = 0.001
    momentum = 0.9
    MIN_WORD_FREQ = 5

    NUM_UNITS_GRU = 150
    BATCH_SIZE = 128
    MAX_SEQ_LEN = 5  # TODO: value?
    EOS = -1

    # Load vocabulary and pre-trained word2vec word vectors.
    WEIGHTS = np.load('data/small_vocab_word_vecs.npy').astype('float32')
    with open('data/small_vocab_word_vocab', 'rb') as f:
        word2vec_vocab = pickle.load(f)
    print "Num word vectors %i" % len(word2vec_vocab)

    # Load list of sentences from OpenSubtitles data (each sentence is a list of words).
    try:
        with open('data/OpenSubtitlesSentences', 'rb') as f:
            sentences = pickle.load(f)
    except IOError:
        sentences = load_sentences(10000)
        pickle.dump(sentences, open('data/OpenSubtitlesSentences', 'wb'))

    # Find the set of words that are in the data (and their frequencies).
    word_set = {}
    for sentence in sentences:
        for w in sentence:
            if w in word2vec_vocab:
                if not w in word_set:
                    word_set[w] = 0
                word_set[w] += 1

    # Only keep words with frequency higher than MIN_WORD_FREQ.
    word_list = [w for w in word_set if word_set[w] > MIN_WORD_FREQ]

    # Only use word2vec vectors that are in the set of words.
    new_W = []
    word_set = {}
    for i, w in enumerate(word_list):
        word_set[w] = i
        new_W.append(WEIGHTS[word2vec_vocab[w], :])

    WEIGHTS = np.vstack(new_W)
    word2vec_vocab = word_set

    word_embedding_size = WEIGHTS.shape[1]
    num_words = WEIGHTS.shape[0]

    # Prepare the data structure that will go into the network (lists of encoded words).
    encoded_sequences = []
    masks = []
    target_vals = []
    for sentence in sentences:
        words = [w for w in sentence if w in word2vec_vocab]
        
        if not words:
            continue

        # Words up to the last word.
        encoded_words, mask = encode_str(words[:-1], word2vec_vocab, MAX_SEQ_LEN)
        encoded_sequences.append(encoded_words)
        masks.append(mask)

        if len(words) > MAX_SEQ_LEN:
            idx = MAX_SEQ_LEN
        else:
            idx = -1

        # Last word.
        pred_word = word2vec_vocab[words[idx]]
        target_vals.append(pred_word)

    encoded_sequences = np.vstack(encoded_sequences).astype('int32')
    masks = np.vstack(masks).astype('int32')

    y = np.vstack(target_vals).astype('int32')

    output_layer, train_func, test_func, predict_func = word_prediction_network(BATCH_SIZE, word_embedding_size, num_words, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU, learning_rate)

    estimator = LasagneNet(output_layer, train_func, test_func, predict_func, on_epoch_finished=[SaveParams('save_params','word_embedding', save_interval = 1)])
    # estimator.draw_network() # requires networkx package

    split_idx = 9000
    X_train = {'X': encoded_sequences[:split_idx], 'X_mask': masks[:split_idx]}
    y_train = y[:split_idx]
    X_test = {'X': encoded_sequences[split_idx:], 'X_mask': masks[split_idx:]}
    #y_test = y[split_idx:]
    
    train = True
    load = False
    test = False
    if train:
        estimator.fit(X_train, y_train)
    if load:
        estimator.load_weights_from('word_embedding/saved_params_42')
    if test:
        predictions = estimator.predict(X_test)
        predictions = predictions.reshape(-1, num_words + 1).argmax(axis=-1)
        word2vec_vocab_rev = dict(zip(word2vec_vocab.values(), word2vec_vocab.keys()))
        for idx in xrange(len(predictions)):
            line = X_test['X'][idx][X_test['X_mask'][idx].astype('bool')]
            print [word2vec_vocab_rev[w] for w in line]
            print word2vec_vocab_rev[predictions[idx]]
        import pdb
        pdb.set_trace()
