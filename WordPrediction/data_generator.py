#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate data for RNN chatbot.
"""

import xml.etree.ElementTree as ET
import gzip
import os
import random
import pickle

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_sentences(num_sents=None):
    '''
    Input:
    num_sents:      Integer, number of sentences to load. By default, as many sentences as are available are loaded.

    Example:
    sentences = load_sentences(num_sents=1000)
    '''
    #data_folder = '/home/olavur/Dropbox/my_folder/DTU/RNN/data/OpenSubtitles/en'
    data_folder = '/zhome/14/2/64409/OpenSubtitles/en'

    data_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(data_folder):
        if filenames:
            for filename in filenames:
                data_filenames.append(dirpath + '/' + filename)

    sentences = []
    for filename in data_filenames:
        file_xml = gzip.open(filename).read()
        root = ET.fromstring(file_xml)
        for sel1 in root.findall('s'):
            sentence = []
            for sel2 in sel1.findall('w'):
                sentence.append(sel2.text.lower())
            sentences.append(sentence)


        # Stop when num_sents reached. Will never happen if num_sents = None (default).
        if num_sents < len(sentences):
            break

    if len(sentences) > num_sents:
        sentences = sentences[:num_sents]

    return sentences

def group_sentences(sentences):
    new_sentences = []
    idx = 0
    while idx < len(sentences) - 1:
        new_sentences.append((sentences[idx], sentences[idx + 1]))
        idx += 1
    return new_sentences

def randomize_sentences(sentences):
    new_sentences = []
    while sentences:
        rand_idx = random.randint(0, len(sentences) - 1)
        rand_sent = sentences.pop(rand_idx)
        new_sentences.append(rand_sent)
    return new_sentences


if __name__ == '__main__':
    NUM_SENTENCES = 500000
    sentences = load_sentences(NUM_SENTENCES)
    logging.info('#Sentences after load: %d', len(sentences))

    grouped_sentences = group_sentences(sentences)
    logging.info('#Sentence pairs after group: %d', len(grouped_sentences))

    grouped_sentences = randomize_sentences(grouped_sentences)
    logging.info('#Sentence pairs after randomize: %d', len(grouped_sentences))

    data = {}
    data['sentences'] = sentences
    data['grouped_sentences'] = grouped_sentences

    pickle.dump(data, open('data/OpenSubtitlesSentences.pickle', 'wb'))
