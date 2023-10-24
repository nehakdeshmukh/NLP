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
import gensim
from gensim.models import LdaModel
import pyLDAvis.gensim
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

bow_doc_300 = corpus[300]

for i in range(len(bow_doc_300)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_300[i][0], 
                                                     dictionary[bow_doc_300[i][0]], 
                                                     bow_doc_300[i][1]))
    
    
def get_lda_topics(model, num__of_topics):
    word_dict = {};
    for i in range(num__of_topics):
        words = model.show_topic(i, topn = 30);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=500,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)

get_lda_topics(lda_model, 15)

pyLDAvis.enable_notebook()
warnings.filterwarnings("ignore", category=DeprecationWarning)

pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

lda_model.save('model15.gensim')
topics = lda_model.print_topics(num_words=6)
for topic in topics:
    print(topic)

# # #evaluation by classifying sample document using LDA Bag of Words model
# for index, score in sorted(lda_model[corpus[300]], key=lambda tup: -1*tup[1]) :
#      print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 14)))

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(topic, idx ))
    print("\n")
    
#Topic Modeling using NMF

data_text = data[['text']]
data_text = data_text.astype('str')

articles = [value[0] for value in data_text.iloc[0:].values]

articles_sentences = [' '.join(text) for text in articles]

vectorizer = CountVectorizer(analyzer='word', max_features=1000);

x_counts = vectorizer.fit_transform(articles);


transformer = TfidfTransformer()
x_tfidf = transformer.fit_transform(x_counts);

xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

num_topics = 10
model = NMF(n_components=num_topics, init='nndsvd');
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    feat_names = vectorizer.get_feature_names_out()
    word_dict = {};    
    for i in range(num_topics):
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        words = [re.sub('\S*@\S*\s?', '', sent) for sent in words]
        words = [re.sub('\s+', ' ', sent) for sent in words]
        words = [re.sub("\'", "", sent) for sent in words]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    return pd.DataFrame(word_dict);

get_nmf_topics(model, 15)


#Cosine Similarity between Documents

words_list=list(chain.from_iterable(docs))

# Cosine similarity
def cosine_sim(text1, text2):
    tfidf_score = TfidfVectorizer().fit_transform([text1, text2])
    return ((tfidf_score * tfidf_score.T).A)[0, 1]

# Most similar article
def closest_doc_name(sentence, docs):
    cos = []
    for i in range(len(docs)):
        cos.append(cosine_sim(', '.join(sentence.split(' ')),', '.join(docs[i])))
    return [titles[x] for x in np.argsort(cos)[-10:][::-1]]

tfidf = TfidfVectorizer(stop_words=stop_words).fit(words_list)

