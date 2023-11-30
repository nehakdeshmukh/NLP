# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:40:37 2023

@author: nehak
"""
import os 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

BASE_DIR = r'C:\Neha\kaggle Projects\Image Captioning'
WORKING_DIR = r'C:\Neha\kaggle Projects\Git hub\NLP\Image captioning using LSTM'

# Load  Model
model = VGG16()

# restructure model
model = Model(inputs = model.inputs , outputs = model.layers[-2].output)

# Summerize
print(model.summary())

