# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:21:06 2024

@author: nehak
"""
import re
import sys
import json
import pickle
import string
import tables
import numpy as np
import pandas as pd
from pandas import json_normalize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer as Stemmer
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from argparse import ArgumentParser

from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)


parser = ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="choose which type of data to create (options are: train, validation & test)")

args = parser.parse_args()


if args.mode == 'train':
    
    file = 'data\\kp20k_training.json'  # TRAIN data

    # Define the file paths and names to save TRAIN data
    x_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed'
    
elif args.mode == 'validation':
    
    file = 'data\\kp20k_validation.json'  

    # Define the file paths and names to save VALIDATION data to tune model parameters
    x_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed'
    
elif args.mode == 'test':
    
    file = 'data\\kp20k_testing.json'  

    # Define the file paths and names to save TEST data to evaluate the final model
    x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed'
    
else:
    print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
    sys.exit()


x_text_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  
y_text_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  


# Define the number of lines to read
batch_size = 256  
max_len = 40  


json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)

for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')

for index, abstract in enumerate(data['abstract']):
    title_abstract = data['title'][index] + '. ' + abstract  
    
    title_abstract = title_abstract.replace('\n', ' ')

    data['abstract'].iat[index] = title_abstract
    
    
    
data['abstract'] = data['abstract'].apply(lambda text: re.sub(r'(?<!\w)([A-Z])\.', r'\1', text.replace('e.g.', 'eg')))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('i.e.', 'ie'))
data['abstract'] = data['abstract'].apply(lambda text: text.replace('etc.', 'etc'))


data['abstract'] = data['abstract'].apply(sent_tokenize)
print(data['abstract'][0])

data = data.explode('abstract')
print(data)

if x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed': 
    data['assemble_documents_index'] = data.index 

data.reset_index(drop=True, inplace=True)


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
                        "you're": "you are", "you've": "you have", "nor": "not", "'s": "s", "s'": "s"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

def replace_contractions(text):
    contractions, contractions_re = get_contractions()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


data['abstract'] = data['abstract'].apply(replace_contractions)

data['keyword'] = data['keyword'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])

def remove_brackets_and_contents(doc):

    ret = ''
    skip1c = 0
    # skip2c = 0
    for i in doc:
        if i == '[':
            skip1c += 1
        # elif i == '(':
        # skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        # elif i == ')'and skip2c > 0:
        # skip2c -= 1
        elif skip1c == 0:  # and skip2c == 0:
            ret += i
    return ret


data['abstract'] = data['abstract'].apply(remove_brackets_and_contents)


newLine_tabs = '\t' + '\n'
newLine_tabs_table = str.maketrans(newLine_tabs, ' ' * len(newLine_tabs))

def remove_references(doc):
    """
    remove references of publications (in document text)
    :param doc: initial text document
    :return: text document without references
    """
    # delete newline and tab characters
    clear_doc = doc.translate(newLine_tabs_table)

    # remove all references of type "Author, J. et al., 2014"
    clear_doc = re.sub(r'[A-Z][a-z]+,\s[A-Z][a-z]*\. et al.,\s\d{4}', "REFPUBL", clear_doc)

    # remove all references of type "Author et al. 1990"
    clear_doc = re.sub("[A-Z][a-z]+ et al. [0-9]{4}", "REFPUBL", clear_doc)

    # remove all references of type "Author et al."
    clear_doc = re.sub("[A-Z][a-z]+ et al.", "REFPUBL", clear_doc)

    return clear_doc


data['abstract'] = data['abstract'].apply(remove_references)

punctuation = string.punctuation  # + '\t' + '\n'
table = str.maketrans(punctuation, ' '*len(punctuation))

def remove_punct_and_non_ascii(text):
    clean_text = text.translate(table)
    clean_text = clean_text.encode("ascii", "ignore").decode() 
    clean_text = re.sub(r"\b[b-zB-Z]\b", "", clean_text)
    return clean_text

def keyword_remove_punct_and_non_ascii(text):
    # remove all single letter except from 'a' and 'A'
    clean_text = [re.sub(r"\b[b-zB-Z]\b", "", keyw.translate(table).encode("ascii", "ignore").decode()) for keyw in text]  # remove non-ascii characters
    return clean_text


data['abstract'] = data['abstract'].apply(remove_punct_and_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_and_non_ascii)
print(data['keyword'])

data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))
print(data['abstract'][3])  
print('convert digits - abstract finish')

print('LEN BEFORE', len(data))
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
data = data[data['keyword'].map(len) > 0]
print('LEN AFTER', len(data))

#tokenizer
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


for index, list_of_keyphrases in enumerate(data['keyword']):
    keyphrases_list = []
    for keyphrase in list_of_keyphrases:  
       
        keyphrase = keyphrase.strip() 
        if len(keyphrase):  
            tokens = word_tokenize(keyphrase)  
            tokens = [tok if not re.match('^\d+$', tok) else 'DIGIT_REPL' for tok in tokens]            
            keyphrases_list.append([Stemmer('porter').stem(keyword.lower()) for keyword in tokens])
    data['keyword'].iat[index] = keyphrases_list

if x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed':  
    data['abstract'].to_csv(x_text_filename, index=False)  
    data[['keyword', 'assemble_documents_index']].to_csv(y_text_filename, index=False)  
    
count_KP = 0 
count_KP_words = 0  
count_NON_KP = 0 
y = [] 
for index, abstract in enumerate(tqdm(data['abstract'])):
    abstract_word_labels = [0] * len(abstract) 

    
    for i, word in enumerate(abstract): 
        for keyphrase in data['keyword'][index]:
            if Stemmer('porter').stem(word) == keyphrase[0]: 
                match_count = 1 
                for j in range(1, len(keyphrase)): 
                    if i + j < len(abstract):
                        if Stemmer('porter').stem(abstract[i + j]) == keyphrase[j]: 
                            match_count += 1
                        else:
                            break  
                    else:
                        break 
                if match_count == len(keyphrase):
                    for x in range(len(keyphrase)):  
                        abstract_word_labels[i + x] = 1
                    count_KP += 1
                    break  
                
        if not abstract_word_labels[i]:  
            count_NON_KP += 1
    count_KP_words += abstract_word_labels.count(1)  
    y.append(abstract_word_labels)

print('KP count: ', count_KP, '\nKP WORDS count: ', count_KP_words, '\nNON-KP count: ', count_NON_KP)

# save data
with open(x_filename+".txt", "w") as fp_x:
    json.dump(X, fp_x)
with open(y_filename+".txt", "w") as fp_y:
    json.dump(y, fp_y)
    
print("Maximum length of title and abstract in the whole dataset", max_len)


for i in tqdm(range(0, len(X), batch_size)):
 
    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=0)

    if not x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed':
        y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)
        y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]
        
    filters = tables.Filters(complib='blosc', complevel=5)
    
    f = tables.open_file(x_filename+'.hdf', 'a')
    ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters,)
    ds[:] = X_batch
    f.close()

    if not x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed': 
        
        f = tables.open_file(y_filename + '.hdf', 'a')
        ds = f.create_carray('/', 'y_data' + str(i), obj=y_batch, filters=filters)
        ds[:] = y_batch
        f.close()
        
    X_batch = None
    y_batch = None
    
    if y_filename == 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed':  
        y_test = pd.DataFrame({'y_test_keyword': y})
        y_test['y_test_keyword'].to_csv(y_filename, index=False) 
    
    with tables.File(x_filename+'.hdf', 'r') as h5f:
        x = h5f.get_node('/x_data'+str(1024)).read() 
        print(x)
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)
    
    if not x_filename == 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed': 
        with tables.File(y_filename+'.hdf', 'r') as h5f:
            y = h5f.get_node('/y_data'+str(1024)).read()  
            print(y)
            print('y SHAPE AFTER', np.array(y, dtype=object).shape)