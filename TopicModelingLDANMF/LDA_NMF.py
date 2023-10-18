# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:01:51 2023

@author: nehak
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from gensim.corpora import Dictionary

data = pd.read_csv(r'articles.csv')

len(data)

data.head(5)

data['length_text'] = data['text'].str.len()

sns.distplot(data['length_text'], color="b")
plt.show()


data['length_title'] = data['title'].str.len()
sns.distplot(data['length_title'], color="r")
plt.show()


titles = [x for x in data['title']]
docs = [x for x in data['text']]


nltk.download('stopwords')


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def list_words(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub(" ",text.lower())
    words = text.split(" ")
    words = [re.sub('\S*@\S*\s?', '', sent) for sent in words] # 
    words = [re.sub('\s+', ' ', sent) for sent in words] # select white space (small s)
    words = [re.sub("\'", "", sent) for sent in words]
    words = [w for w in words if not len(w) < 2]
    words = [w for w in words if w not in stop_words]
    words = [lem.lemmatize(w) for w in words]

    return words 


nltk.download('wordnet')
docs = [list_words(x) for x in data['text']]

dictionary = Dictionary(docs)
print('Number of unique words in initital documents:', len(dictionary))


dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Number of unique words after removing rare and common words:', len(dictionary))

corpus = [dictionary.doc2bow(doc) for doc in docs]
print(len(corpus))
