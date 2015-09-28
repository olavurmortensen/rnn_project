from __future__ import absolute_import

from LasagneNet import *
from RepeatLayer import RepeatLayer
from os.path import join
import os
from collections import OrderedDict
import itertools
from warnings import warn
from time import time
from lasagne.layers import InputLayer
import lasagne
import theano.tensor as T
import numpy as np
import theano
from SimpleDatabaseIterator import SimpleDatabaseIterator
from sklearn.cross_validation import train_test_split
import codecs
import cPickle as pickle
import gzip
#from SolrSearchModel import *
import nltk

umls_db_name = "UMLS2015"
user = 'root'
password = 'findzebra'
umls_db_host = '62.61.146.181'


def encode_str(text, token_dict, max_seq_len):
    text = ([c for c in text if c in token_dict])
    text = text[0:max_seq_len]
    enc_str = [token_dict[c] for c in text]
    remainder = max_seq_len-len(text)
    enc_str = np.pad(enc_str, ((0, remainder)), mode='constant', constant_values=(((0, 0))))
    mask = np.asarray([1]*len(text) + [0]*remainder, dtype=np.float32)
    return enc_str, mask

def word_prediction_network(BATCH_SIZE, EMBEDDING_SIZE, NUM_WORDS, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU):
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

    l_gru = lasagne.layers.GRULayer(l_emb, num_units=NUM_UNITS_GRU, name='gru', mask_input=l_mask)
    print "gru shape: %s" % str(lasagne.layers.get_output(l_gru, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)

    # slice last index of dimension 1
    l_last_hid = lasagne.layers.SliceLayer(l_gru, indices=-1, axis=1, name='l_last_hid')
    print  "l_last_hid shape: %s" % str(lasagne.layers.get_output(l_last_hid, inputs={l_in: x_sym, l_mask: xmask_sym}).eval(
        {x_sym: X, xmask_sym: Xmask}).shape)

    l_softmax = lasagne.layers.DenseLayer(l_last_hid, num_units=NUM_OUTPUTS,
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
    eq = T.eq(argmax, y_sym)
    acc = T.mean(eq)  #accuracy

    all_trainable_parameters = lasagne.layers.get_all_params([l_softmax], trainable=True)

    #add grad clipping to avoid exploding gradients
    all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_trainable_parameters)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

    updates = lasagne.updates.adam(all_grads, all_trainable_parameters, learning_rate=0.005) # adaptive learning rate should be implemented...

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
    NUM_DOCS = 3

    WEIGHTS = np.load('small_vocab_word_vecs.npy').astype('float32')
    with open('small_vocab_word_vocab', 'rb') as f:
        word2vec_vocab = pickle.load(f)
    print "Num word vectors %i" % len(word2vec_vocab)

    # from gensim.models import word2vec
    # model = word2vec.Word2Vec.load_word2vec_format('C:/data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    # word2vec_vocab = model.vocab
    # word2vec_small = model.syn0


    # build network
    BATCH_SIZE = 128
    NUM_UNITS_GRU = 150
    NUM_SAMPLES_PR_ARTICLE = 500
    MAX_SEQ_LEN = 5
    EOS = -1


    # NUM_UNITS_ENC_WORD = 250

    char_dict = {}
    word_set = {}
    sql = 'select p.text, p.display_title FROM page p, watson_questions where answer = p.display_title and length(p.text)>150 limit 0,%i'%NUM_DOCS
    # sql = "select p.text, p.display_title FROM page p, watson_questions where answer = p.display_title and length(p.text)>150 limit 0,50000"
    db_iter = SimpleDatabaseIterator(umls_db_host, 'findzebra2', user, password, sql) # laptop
    for row in db_iter:
        # print row['display_title']
        text = row['text'].lower()
        words = nltk.word_tokenize(text)
        for w in words:
            if w in word2vec_vocab:
                if w not in word_set:
                    word_set[w] = 0
                word_set[w] += 1

    word_list = [w for w in word_set if word_set[w] > 5]
    word_set = {}

    new_W = []
    for i, w in enumerate(word_list):
        word_set[w] = i
        new_W.append(WEIGHTS[word2vec_vocab[w],:])

    WEIGHTS = np.vstack(new_W)
    word2vec_vocab = word_set

    word_embedding_size = WEIGHTS.shape[1]
    num_words = WEIGHTS.shape[0]


    """
    Generate training data
    """
    encoded_sequences_word = []
    masks_word = []
    sentences = {}
    target_vals = []
    db_iter = SimpleDatabaseIterator(umls_db_host, 'findzebra2', user, password, sql) # laptop
    for row in db_iter:
        text = row['text'].lower()
        title = row['display_title'].lower()
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            idx = range(0, len(sent)-6)
            words = nltk.word_tokenize(sent)
            if len(words) < 8:
                continue
            # print ' '.join(words)
            words = [w for w in words if w in word2vec_vocab]
            # print ' '.join(words)
            for i in idx:
                encoded_string_words, mask_words = encode_str(words[i:(i+3)], word2vec_vocab, MAX_SEQ_LEN)
                encoded_sequences_word.append(encoded_string_words)
                masks_word.append(mask_words)


                xx = [word2vec_vocab[w] for w in words[(i+3):(i+4)] if w in word2vec_vocab]+[EOS]
                xx = xx + [EOS]*(MAX_SEQ_LEN+1-len(xx))
                target_vals.append(xx)


    print "Num samples %i" % len(masks_word)

    encoded_sequences_word = np.vstack(encoded_sequences_word).astype('int32')
    masks_word = np.vstack(masks_word).astype('int32')

    y = np.vstack(target_vals).astype('int32')



    output_layer, train_func, test_func, predict_func = word_prediction_network(BATCH_SIZE, word_embedding_size, num_words, MAX_SEQ_LEN, WEIGHTS, NUM_UNITS_GRU)


    estimator = LasagneNet(output_layer, train_func, test_func, predict_func, on_epoch_finished=[SaveParams('save_params','word_embedding', save_interval = 1)])
    # estimator.draw_network() # requires networkx package


    X = {'X': encoded_sequences_word, 'X_mask': masks_word }

    estimator.fit(X, y)





