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
import sys
from time import time
import lasagne
import theano.tensor as T
import numpy as np
import theano
import cPickle as pickle
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

    #l_gru = lasagne.layers.LSTMLayer(l_emb, num_units=NUM_UNITS_GRU, name='gru', mask_input=l_mask)
    #print "gru shape: %s" % str(lasagne.layers.get_output(l_gru, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
    #    {x_sym: X, xmask_sym: Xmask}).shape)

    ## slice last index of dimension 1
    #l_last_hid = lasagne.layers.SliceLayer(l_gru, indices=-1, axis=1, name='l_last_hid')
    #print  "l_last_hid shape: %s" % str(lasagne.layers.get_output(l_last_hid, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
    #    {x_sym: X, xmask_sym: Xmask}).shape)

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
    try:
        learning_rate = float(sys.argv[1])
    except IndexError:
        learning_rate = 0.0001
    momentum = 0.9
    MIN_WORD_FREQ = 5
    train_split = 1000
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
    with open('../data/OpenSubtitlesSentences.pickle', 'rb') as f:
        data = pickle.load(f)
    sentences = data['sentences']
    sent_pairs = data['grouped_sentences']
    train_sent_pairs = sent_pairs[:train_split]
    test_sent_pairs = sent_pairs[train_split:test_split]

    logging.info('Training examples: %d', len(train_sent_pairs))
    logging.info('Test examples: %d', len(test_sent_pairs))
    
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

    ## Prepare the data structure that will go into the network (lists of encoded words). ##

    # Training data.
    encoded_sequences = []
    masks = []
    target_vals = []
    for query, response in train_sent_pairs:
        # Remove out-of-vocabulary words.
        query = [w for w in query if w in word2vec_vocab]
        response = [w for w in response if w in word2vec_vocab]
        
        # If query or response is empty after OOV words are removed, we continue.
        if not query or not response:
            continue

        # Training examples.
        for i in range(len(response)):
            # Input sequence.
            if i == 0:
                input_seq = query
            else:
                input_seq = query + response[:i]
            encoded_words, mask = encode_str(input_seq, word2vec_vocab, MAX_SEQ_LEN)
            encoded_sequences.append(encoded_words)
            masks.append(mask)

            # Output word.
            pred_word = word2vec_vocab[response[i]]
            target_vals.append(pred_word)

            # If sequence exceeds maximum length, continue to next sentence pair.
            if len(input_seq) >= MAX_SEQ_LEN:
                break

    encoded_sequences = np.vstack(encoded_sequences).astype('int32')
    masks = np.vstack(masks).astype('float32')

    X_train = {'X': encoded_sequences, 'X_mask': masks}
    y_train = np.vstack(target_vals).astype('int32')

    # Test data.
    encoded_sequences = []
    masks = []
    target_vals = []
    for query, response in test_sent_pairs:
        # Remove out-of-vocabulary words.
        query = [w for w in query if w in word2vec_vocab]
        response = [w for w in response if w in word2vec_vocab]
        
        if not query or not response:
            continue

        # If query sequence exceeds maximum length, continue to next sentence pair.
        if len(query) > MAX_SEQ_LEN:
            continue

        encoded_words, mask = encode_str(query, word2vec_vocab, MAX_SEQ_LEN)
        encoded_sequences.append(encoded_words)
        masks.append(mask)

        # TODO: use response also, so I can see the difference between predicted and actual.

    encoded_sequences = np.vstack(encoded_sequences).astype('int32')
    masks = np.vstack(masks).astype('float32')

    X_test = {'X': encoded_sequences, 'X_mask': masks}

    output_layer, train_func, test_func, predict_func = word_prediction_network(BATCH_SIZE, word_embedding_size, num_words, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU, learning_rate)

    estimator = LasagneNet(output_layer, train_func, test_func, predict_func, on_epoch_finished=[SaveParams('save_params','word_embedding', save_interval = 1)])
    # estimator.draw_network() # requires networkx package

    train = False
    if train:
        estimator.fit(X_train, y_train)
    else:
        estimator.load_weights_from('word_embedding/saved_params_3')
        pred_sents = []
        # For each test example, predict the response.
        for idx in xrange(X_test['X'].shape[0]):
            pred_words = []

            temp = X_test['X'][idx].reshape(1, MAX_SEQ_LEN)
            X_new = np.empty_like(temp)
            X_new[:] = temp

            temp = X_test['X_mask'][idx].reshape(1, MAX_SEQ_LEN)
            X_mask_new = np.empty_like(temp)
            X_mask_new[:] = temp
            # Predict one word at a time, based on the previous predicted words and the query.
            for pred_iter in range(10):  # FIXME: predicting 10 words. To stop predicting, make the model predict <EOS>.
                zero_found = False
                prediction = estimator.predict({'X': X_new, 'X_mask': X_mask_new})  # Predict current sequence.
                prediction = prediction.reshape(-1, num_words + 1).argmax(axis=-1)[0]  # Take the argmax of the softmax output.
                pred_words.append(prediction)
                # Find where to place the prediction in X and X_mask.
                for i, elem in enumerate(X_mask_new[0]):
                    if elem == 0:
                        zero_found = True
                        X_mask_new[0, i] = 1
                        X_new[0, i] = prediction
                        break
                if zero_found == False:
                    # Can't predict more because of MAX_SEQ_LEN.
                    break
            pred_sents.append(pred_words)

        # Find words corresponding to the integer tokens in the query and predicted response, and print them.
        word2vec_vocab_rev = dict(zip(word2vec_vocab.values(), word2vec_vocab.keys()))
        for idx in xrange(len(pred_sents)):
            query = X_test['X'][idx][X_test['X_mask'][idx].astype('bool')]  # Apply mask to query.
            print 'Query: %r' % [word2vec_vocab_rev[w] for w in query]
            response = [word2vec_vocab_rev[w] for w in pred_sents[idx]]
            print 'Response: %r' % response
        import pdb
        pdb.set_trace()
