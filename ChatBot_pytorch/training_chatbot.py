# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:32:44 2023

@author: nehak
"""

import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import json
import numpy as np 

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open(r"intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
ignore_words = ["?",".","!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(tags)
        
x_train = []
y_train = []        

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) # cross entropy loss 
    
x_train = np.array(x_train)
y_train = np.array(y_train)
        
    
# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000


print(input_size, output_size)


class Chat_dataset(Dataset):
    
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train 
        
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    
    def __len__(self):
        return self.n_samples

dataset = Chat_dataset()

train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)
        
