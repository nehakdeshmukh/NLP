# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:45:28 2023

@author: nehak
"""

import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential


data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\NLP\Sentiment.csv")

data = data[['text','sentiment']]

data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)


for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

