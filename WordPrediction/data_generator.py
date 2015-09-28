#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate data for RNN chatbot.
"""

import xml.etree.ElementTree as ET
import gzip
import os

def load_sentences(num_sents=None):
    '''
    Input:
    num_sents:      Integer, number of sentences to load. By default, as many sentences as are available are loaded.

    Example:
    sentences = load_sentences(num_sents=1000)
    '''
    data_folder = '/home/olavur/Dropbox/my_folder/DTU/RNN/data/OpenSubtitles/en'

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
                sentence.append(sel2.text)
            sentences.append(sentence)


        # Stop when num_sents reached. Will never happen if num_sents = None (default).
        if num_sents < len(sentences):
            break

    if len(sentences) > num_sents:
        sentences = sentences[:num_sents]

    return sentences
