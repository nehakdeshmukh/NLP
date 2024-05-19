# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:18:37 2024

@author: nehak
"""
import json
import pandas as pd
from pandas import json_normalize
import re
import string 
from keras.preprocessing.text import Tokenizer
import pickle 
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer
import numpy as np


file = 'data\\kp20k_validation.json' 



# VALIDATION data path
x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'



#  TEST data - use for EVALUATION (exact/partial matching)
x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  
y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  


batch_size = 32  

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)

#Split keyphrases list
for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')  

for index, abstract in enumerate(data['abstract']):
    title_abstract = data['title'][index] + '. ' + abstract  # title + abstract
    # remove '\n'
    title_abstract = title_abstract.replace('\n', ' ')

    data['abstract'].iat[index] = title_abstract
    
def get_contractions():
    contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                        "could've": "could have", "couldn't": "could not", "didn't": "did not",
                        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                        "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                        "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                        "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                        "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                        "she'll've": "she will have", "she's": "she is", "should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                        "so's": "so as", "this's": "this is", "that'd": "that would",
                        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                        "they'll've": "they will have", "they're": "they are", "they've": "they have",
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                        "when've": "when have", "where'd": "where did", "where's": "where is",
                        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                        "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have", "nor": "not"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


def replace_contractions(text):
    contractions, contractions_re = get_contractions()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


print('BEFORE contractions data[abstract]', data['abstract'])
data['abstract'] = data['abstract'].apply(replace_contractions)
print('AFTER contractions data[abstract]', data['abstract'])

print('BEFORE contractions data[keyword]', data['keyword'])
data['keyword'] = data['keyword'].apply(
    lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])
print('AFTER contractions data[keyword]', data['keyword'])

punctuation = string.punctuation + '\t' + '\n'
punctuation = punctuation.replace("'", '')  
table = str.maketrans(punctuation, ' ' * len(punctuation))
print(punctuation, 'LEN:', len(punctuation))

def remove_punct_and_non_ascii(text):
    clean_text = text.translate(table)
    clean_text = clean_text.encode("ascii", "ignore").decode()  # remove non-ascii characters
    return clean_text

def keyword_remove_punct_and_non_ascii(text):
    clean_text = [keyw.translate(table).encode("ascii", "ignore").decode() for keyw in text]  # remove non-ascii characters
    return clean_text

# remove punctuation
data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)
print(data['keyword'])

data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))

data = data[data['abstract'].str.strip().astype(bool)]

data.reset_index(drop=True, inplace=True)
print('AFTER CLEANING', data)

print('LEN BEFORE', len(data))
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])

data = data[data['keyword'].map(len) > 0]
print('LEN AFTER', len(data))

if x_filename == 'data\\preprocessed_data\\x_TRAIN_data_preprocessed':  

    tokenizer = Tokenizer(filters='', lower=True, oov_token='<UKN>')
    tokenizer.fit_on_texts(data['abstract'])
    
    # convert text to sequence of numbers
    X = tokenizer.texts_to_sequences(data['abstract'])
    
    # word-index pairs
    word_index = tokenizer.word_index
    
    with open('data\\train_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:  
    
    # load tokenizer
    with open('data\\train_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    X = tokenizer.texts_to_sequences(data['abstract'])
    
    word_index = tokenizer.word_index

def tokenize_lowercase(text):

    formatted_text = []
    words = word_tokenize(text) 
    for word_token in words:  
        formatted_text.append(word_token.lower())  
    return formatted_text

# tokenize text
data['abstract'] = data['abstract'].apply(tokenize_lowercase)
print(data['abstract'])
print('tokenization - abstract finish')

for index, list_of_keyphrases in enumerate(data['keyword']):
    keyphrases_list = []
    for keyphrase in list_of_keyphrases: 
       
        keyphrase = keyphrase.strip() 
        if len(keyphrase): 
            tokens = word_tokenize(keyphrase) 
         
            tokens = [tok if not re.match('^\d+$', tok) else 'DIGIT_REPL' for tok in tokens]
           
            keyphrases_list.append([Stemmer('porter').stem(keyword.lower()) for keyword in tokens])  # stem + lower case
    data['keyword'].iat[index] = keyphrases_list
    
if x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':  
    data['abstract'].to_csv(x_text_filename, index=False) 
    data['keyword'].to_csv(y_text_filename, index=False) 
    
    
if x_filename == 'data\\preprocessed_data\\x_TRAIN_data_preprocessed':

    gloveFile = 'GloVe\\glove.6B\\glove.6B.100d.txt'
    print("Loading Glove Model")
    
    
    glove_model = {}
    with open(gloveFile, 'r', encoding="utf8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array(splitLine[1:], dtype='float32')
            glove_model[word] = embedding

    print("Found %s word vectors." % len(glove_model))