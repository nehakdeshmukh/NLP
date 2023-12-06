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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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
    
    image_id = image_id.split('.')[0]
    
    caption = " ".join(caption)
    
    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append(caption)
    
    
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            #one caption at a time
            caption = captions[i]
  
            # lowercase
            caption = caption.lower()
            
            caption = caption.replace('[^A-Za-z]', '')
            
            caption = caption.replace('\s+', ' ')
            
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            
            captions[i] = caption
            
            
mapping['1000268201_693b08cb0e']
            
clean(mapping)

mapping['1000268201_693b08cb0e']

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

all_captions[:15]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

max_length = max(len(caption.split()) for caption in all_captions)
max_length

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


def data_generator(data_keys, mapping, tokenizer):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    for key in data_keys:
        n += 1
        captions = mapping[key]
        
        for caption in captions:
           
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                # split into input and output pairs
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                
                out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
                
                X1.append(features[key][0])
                X2.append(in_seq)
                y.append(out_seq)