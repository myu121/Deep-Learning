#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 19:26:17 2019

@author: miaoyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import itertools
import datetime
import csv 

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# Setting Path
#path = '/Users/miaoyu/Dropbox/ST790--001/quora-question-pairs'
#os.chdir(path)
data = pd.read_csv("processed_data.csv")

import spacy
nlp = spacy.load('en_core_web_lg-2.2.5')

vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
questions_cols = ['question1', 'question2']
integ_data = data.copy()

for index, row in integ_data.iterrows():
    
    if index % 100 == 0:
        print(index)
    
    for question in questions_cols:

        q2n = []  # q2n -> question numbers representation
        for word in row[question].split():
            
            if word not in list(nlp.vocab.strings):
                continue
            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                q2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                q2n.append(vocabulary[word])

        # Replace questions as word to question as number representation
        integ_data.set_value(index, question, q2n)
        

df_vocabulary = pd.DataFrame(list(vocabulary.items()), columns=['word', 'word_index'])
df_inv_vocabulary = pd.DataFrame(inverse_vocabulary, columns=['word'])

integ_data.to_csv("integ_data.csv")
df_vocabulary.to_csv('vocabulary.csv')
df_inv_vocabulary.to_csv('inv_vocabulary.csv')