# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:40:37 2023

@author: nehak
"""
import os 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm 
from tensorflow.keras.preprocessing.image import load_img



BASE_DIR = r'C:\Neha\kaggle Projects\Image Captioning'
WORKING_DIR = r'C:\Neha\kaggle Projects\Git hub\NLP\Image captioning using LSTM'

# Load  Model
model = VGG16()

# restructure model
model = Model(inputs = model.inputs , outputs = model.layers[-2].output)

# Summerize
print(model.summary())

# extract features from image
features = {}
directory = os.path.join(BASE_DIR, 'Images')


for img_name in tqdm(os.listdir(directory)):
    # load the image from file
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    print(image)