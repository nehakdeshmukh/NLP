# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:22:00 2023

@author: nehak
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf 
import random
import pickle 

from tensorflow.python.framework import ops
ops.reset_default_graph()


import json
with open(r'intents.json') as file:
    data = json.load(file)
    
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    
    
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        
        
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    training = []
    
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words :
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)
    
    