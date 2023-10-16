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


