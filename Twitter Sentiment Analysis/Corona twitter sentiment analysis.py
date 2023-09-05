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
import re,string
import emoji


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


tweets_per_region = train_data['Location'].value_counts().loc[lambda x : x > 50].reset_index(name='counts')

plt.figure(figsize=(15,6))
ax = sns.barplot(x='index', y='counts', data=tweets_per_region,edgecolor = 'black',ci=False, palette='Spectral')
plt.title('Tweets count by region')
plt.xticks(rotation=70)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


df = train_data[['OriginalTweet','Sentiment']]

df_test = test_data[['OriginalTweet','Sentiment']]


#remove emojis
def strip_emoji(text):
    # return emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))
    return emoji.replace_emoji(text, replace='')


def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r & make it lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters
    
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text 

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)


def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence

    return new_tweet2


#Filter special characters 
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)



texts_new = []
for t in df.OriginalTweet:
    texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))
    
    
texts_new_test = []
for t in df_test.OriginalTweet:
    texts_new_test.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))))
    
    
df['text_clean'] = texts_new
df_test['text_clean'] = texts_new_test


df['text_clean'].head()
df_test['text_clean'].head()


df['text_clean'][1:8].values

text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)