#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Sequence learning using LasagneNet.
'''

from __future__ import absolute_import

from LasagneNet import *
from RepeatLayer import RepeatLayer
import os
from time import time
import lasagne
import theano.tensor as T
import numpy as np
import theano
import cPickle as pickle
from math import ceil
import sys

sys.stdout = sys.stderr

import pdb

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

    l_in = lasagne.layers.InputLayer((BATCH_SIZE, MAX_SEQ_LEN), name='input')  # TODO: BATCH_SIZE instead of "None"?
    print "l_in shape: %s" % str((lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: X}).shape))

    l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_WORDS, EMBEDDING_SIZE,
                                              W=WEIGHTS.astype('float32'),
                                              name='embedding')

    l_emb.params[l_emb.W].remove('trainable')
    print "l_emb shape: %s" % str((lasagne.layers.get_output(l_emb, inputs={l_in: x_sym}).eval({x_sym: X}).shape))

    l_mask = lasagne.layers.InputLayer((BATCH_SIZE, MAX_SEQ_LEN), name='input_mask')

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

    # Sample words from the softmax layer.
    #rng =  T.raw_random.random_state_type()  # Random state.
    #new_rng, sample = T.raw_random.multinomial(rng, pvals=output_train)  # Sample from the softmax layer.
    #sample = T.argmax(sample, axis=1)  # Return the index of the item that was sampled.
    #eq = T.eq(sample, y_sym.flatten())
    #acc = T.mean(eq)  #accuracy

    all_trainable_parameters = lasagne.layers.get_all_params([l_softmax], trainable=True)

    #add grad clipping to avoid exploding gradients
    clip_level = 10
    all_grads = [T.clip(g, -clip_level, clip_level) for g in T.grad(mean_cost, all_trainable_parameters)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads, clip_level)

    updates = lasagne.updates.adam(
            all_grads,
            all_trainable_parameters,
            learning_rate=learning_rate,) # adaptive learning rate should be implemented...

    train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc], updates=updates)
    test_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc])
    predict_func = theano.function([x_sym, xmask_sym], [output_train])

    hidden_repr = lasagne.layers.get_output(l_concat,
            inputs={l_in: x_sym, l_mask: xmask_sym},
            deterministic=False)

    get_hidden_func = theano.function([x_sym, xmask_sym], [hidden_repr])

    # when the input X is a dict, the following definitions will allow LasagneNet to call train_func without
    # knowing the order of the inputs, using the syntax train_function(**X)
    def train_function(X, y, X_mask):
        return train_func(X, y, X_mask)

    def test_function(X, y, X_mask):
        return test_func(X, y, X_mask)

    def predict_function(X, X_mask):
        return predict_func(X, X_mask)

    def get_hidden_function(X, X_mask):
        return get_hidden_func(X, X_mask)

    return l_softmax, train_function, test_function, predict_function, get_hidden_function


if __name__ == "__main__":
    try:
        learning_rate = float(sys.argv[1])
    except IndexError:
        learning_rate = 0.0001
    momentum = 0.9  # NOTE: not used a.t.m.
    MIN_WORD_FREQ = 5
    min_train_sent_len = 3
    train_split = 100000
    test_split = train_split + 100

    NUM_UNITS_GRU = 500
    BATCH_SIZE = 128
    MAX_SEQ_LEN = 10
    EOS = -1

    # Load vocabulary and pre-trained word2vec word vectors.
    #WEIGHTS = np.load('data/word_embeddings_vecs.pickle').astype('float32')
    with open('../data/word_embeddings_vecs.pickle', 'rb') as f:
        WEIGHTS = pickle.load(f)
    with open('../data/word_embeddings_vocab.pickle', 'rb') as f:
        word2vec_vocab = pickle.load(f)
    print "Num word vectors %i" % len(word2vec_vocab)

    # Load data.
    data = pickle.load(open('../data/OpenSubtitlesSentences.pickle', 'rb'))
    sentences = data['sentences']

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
        
        if len(words) < min_train_sent_len:
            continue

        pred_idx = int(ceil(len(words)/float(2)))  # First word to predict: the middle, rounding up.
        for i in range(pred_idx, len(words)):
            # Words to predict based on.
            encoded_words, mask = encode_str(words[:i], word2vec_vocab, MAX_SEQ_LEN)
            encoded_sequences.append(encoded_words)
            masks.append(mask)

            # Word to predict.
            pred_word = word2vec_vocab[words[i]]
            target_vals.append(pred_word)
            
            if i - 1 == MAX_SEQ_LEN:
                break

    encoded_sequences = np.vstack(encoded_sequences).astype('int32')
    masks = np.vstack(masks).astype('float32')

    y = np.vstack(target_vals).astype('int32')

    output_layer, train_func, test_func, predict_func, get_hidden_func = word_prediction_network(BATCH_SIZE, word_embedding_size, num_words, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU, learning_rate)

    estimator = LasagneNet(output_layer, train_func, test_func, predict_func, get_hidden_func, on_epoch_finished=[SaveParams('save_params','word_embedding', save_interval = 1)])
    # estimator.draw_network() # requires networkx package

    X_train = {'X': encoded_sequences[:train_split], 'X_mask': masks[:train_split]}
    y_train = y[:train_split]
    X_test = {'X': encoded_sequences[train_split:test_split], 'X_mask': masks[train_split:test_split]}
    y_test = y[train_split:test_split]
    
    train = False
    if train:
        estimator.fit(X_train, y_train)
    else:
        estimator.load_weights_from('saved_params')
        word2vec_vocab_rev = dict(zip(word2vec_vocab.values(), word2vec_vocab.keys()))  # Maps indeces to words.

        predictions = estimator.predict(X_test)
        predictions = predictions.reshape(-1, num_words + 1)  # Reshape into #samples x #words.
        n_pred = predictions.shape[0]
        for row in range(n_pred):
            # Sample a word from output probabilities.
            sample = np.random.multinomial(n=1, pvals=predictions[row,:]).argmax()
            line = X_test['X'][row]  # Get the input line.
            line = line[X_test['X_mask'][row].astype('bool')]  # Apply mask.
            print 'Input: %r' %([word2vec_vocab_rev[w] for w in line])  # Print words in line.
            print 'Guess:: %r' %(word2vec_vocab_rev[sample])  # Print predicted word.
            print 'Output: %r' %(word2vec_vocab_rev[y_test[row]])  # Print the correct word.
        pdb.set_trace()
