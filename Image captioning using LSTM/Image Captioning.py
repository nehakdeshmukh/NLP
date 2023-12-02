# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:40:37 2023

@author: nehak
"""
import os 
from tensorflow.keras.applications.vgg16 import VGG16,  preprocess_input
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle 


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
    # print(image)
    
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = model.predict(image, verbose=0)
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature
    
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
    
    
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]