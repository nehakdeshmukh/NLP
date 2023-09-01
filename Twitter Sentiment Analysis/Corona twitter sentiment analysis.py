# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:29:40 2023

@author: nehak
"""

import pandas as pd
import numpy as np 
import plotly.graph_objects as go
from plotly.offline import plot
import seaborn  as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\NLP\Twitter Sentiment Analysis\Corona_NLP_train.csv",encoding='ISO-8859-1')
 
test_data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\NLP\Twitter Sentiment Analysis\Corona_NLP_test.csv")

train_data.head()

train_data.info()

train_data['TweetAt'] = pd.to_datetime(train_data['TweetAt'])

train_data.drop_duplicates(subset='OriginalTweet',inplace=True)


tweets_per_day = train_data['TweetAt'].dt.strftime('%m-%d').value_counts().sort_index().reset_index(name='counts')
tweets_per_day = tweets_per_day.rename(columns={'index': 'Date'})



plt.figure(figsize=(20,5))
ax = sns.barplot(x='Date', y='counts', data=tweets_per_day,edgecolor = 'black',ci=False, palette='Blues_r')
plt.title('Tweets count by date')
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


tweets_per_country = train_data['Location'].value_counts().loc[lambda x : x > 100].reset_index(name='counts')
