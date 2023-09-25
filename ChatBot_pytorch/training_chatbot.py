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