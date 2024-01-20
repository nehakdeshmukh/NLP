# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:53:09 2024

@author: nehak
"""

#preprocessing_full

import re
import sys
import json
import pickle
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pandas import json_normalize
import string
from nltk.tokenize import word_tokenize
import operator
import pad_sequences
import to_categorical
import tables
from numpy import savez_compressed

from tqdm import tqdm
tqdm.pandas()
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.snowball import SnowballStemmer as Stemmer

pd.set_option('display.max_columns', None)

parser = ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="choose which type of data to create (options are: train, validation or test)")

args = parser.parse_args()


if args.mode == 'train':
    # reading the initial JSON data using json.load()
    file = 'data\\kp20k_training.json'  # TRAIN data

    # Define the file paths and names to save TRAIN data
    x_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed'
elif args.mode == 'validation':
   
    file = 'data\\kp20k_validation.json'  

    # Define the file paths and names to save VALIDATION data 
    x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'
elif args.mode == 'test':
    
    file = 'data\\kp20k_testing.json'  
    # Define the file paths and names to save TEST data 
    x_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'

else:
    print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
    sys.exit()

    x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT' 
    y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT' 


batch_size = 64  
max_len = 400 


json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

#split keyphrases 
for index, keywords in enumerate(data['keyword']):
    data['keyword'].iat[index] = keywords.split(';')
    
    
for index, abstract in enumerate(data['abstract']):
    title_abstract =  data['title'][index] + '. ' + abstract  # combine title + abstract
    
    title_abstract = title_abstract.replace('\n', ' ')

    data['abstract'].iat[index] = title_abstract
    
def contractions_func():
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
    contractions, contractions_re = contractions_func()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


print('BEFORE contractions data[abstract]', data['abstract'])
data['abstract'] = data['abstract'].apply(replace_contractions)
print('AFTER contractions data[abstract]', data['abstract'])

print('BEFORE contractions data[keyword]', data['keyword'])
data['keyword'] = data['keyword'].apply(lambda set_of_keyphrases: [replace_contractions(keyphrase) for keyphrase in set_of_keyphrases])
print('AFTER contractions data[keyword]', data['keyword'])


def remove_brackets_contents(doc):
    """
    remove parenthesis, brackets and their contents
    :param doc: initial text document
    :return: text document without parenthesis, brackets and their contents
    """
    ret = ''
    skip1c = 0
    
    for i in doc:
        if i == '[':
            skip1c += 1
        
        elif i == ']' and skip1c > 0:
            skip1c -= 1
            
        elif skip1c == 0: 
            ret += i
    return ret


# remove parenthesis, brackets, contents
data['abstract'] = data['abstract'].apply(remove_brackets_contents)


# delete newline, tab characters
newLine_tabs = '\t' + '\n'
newLine_tabs_table = str.maketrans(newLine_tabs, ' ' * len(newLine_tabs))
print(newLine_tabs, 'newLine_tabs LEN:', len(newLine_tabs))

def remove_references(doc):
    
    clear_doc = doc.translate(newLine_tabs_table)    
    clear_doc = re.sub(r'[A-Z][a-z]+,\s[A-Z][a-z]*\. et al.,\s\d{4}', "REFPUBL", clear_doc)  
    clear_doc = re.sub("[A-Z][a-z]+ et al. [0-9]{4}", "REFPUBL", clear_doc)
    clear_doc = re.sub("[A-Z][a-z]+ et al.", "REFPUBL", clear_doc)

    return clear_doc


# remove references in documents 
data['abstract'] = data['abstract'].apply(remove_references)

punctuation = string.punctuation
table = str.maketrans(punctuation, ' '*len(punctuation))
print(punctuation, 'LEN:', len(punctuation))

def remove_punct_non_ascii(text):
    clean_text = text.translate(table)
    clean_text = clean_text.encode("ascii", "ignore").decode()  
    clean_text = re.sub(r"\b[b-zB-Z]\b", "", clean_text)
    return clean_text

def keyword_remove_punct_non_ascii(text):
    clean_text = [re.sub(r"\b[b-zB-Z]\b", "", keyw.translate(table).encode("ascii", "ignore").decode()) for keyw in text]  # remove non-ascii characters
    return clean_text

# remove punctuation
data['abstract'] = data['abstract'].apply(remove_punct_non_ascii)
data['keyword'] = data['keyword'].apply(keyword_remove_punct_non_ascii)

# remove spaces
data['abstract'] = data['abstract'].apply(lambda text: " ".join([token if not re.match('^\d+$', token) else 'DIGIT_REPL' for token in text.split()]))  

# remove empty sentences
data = data[data['abstract'].str.strip().astype(bool)]
# reset index 
data.reset_index(drop=True, inplace=True)

# remove empty keyphrases
data['keyword'] = data['keyword'].apply(lambda set_of_keyws: [key_text for key_text in set_of_keyws if key_text.strip()])
# remove rows keyphrases
data = data[data['keyword'].map(len) > 0]

#create the GloVe matrix, that will be used as weights in the embedding layer
if x_filename == 'data\\preprocessed_data\\x_TRAIN_data_preprocessed': 

    # oov_token
    tokenizer = Tokenizer(filters='', lower=True, oov_token='<UKN>')
    tokenizer.fit_on_texts(data['abstract'])

    # convert text to sequence of numbers
    X = tokenizer.texts_to_sequences(data['abstract'])

    # get the word-index pairs
    word_index = tokenizer.word_index

    # save tokenizer
    with open('data\\train_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

else:  # for validation and test sets

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

# stem, tokenize and lower case keyphrases 
for index, list_of_keyphrases in enumerate(data['keyword']):
    keyphrases_list = []
    for keyphrase in list_of_keyphrases:  
        keyphrase = keyphrase.strip()  
        if len(keyphrase):  
            tokens = word_tokenize(keyphrase)  
            tokens = [tok if not re.match('^\d+$', tok) else 'DIGIT_REPL' for tok in tokens]
            keyphrases_list.append([Stemmer('porter').stem(keyword.lower()) for keyword in tokens])
    data['keyword'].iat[index] = keyphrases_list

if x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':  
    data['abstract'].to_csv(x_text_filename, index=False)  
    data['keyword'].to_csv(y_text_filename, index=False)  
    
# Glove metrix
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
    
def get_glove_vec(word_vec):
    """
    Get the vector of a word
    """
    embedding_vector = glove_model.get(word_vec)  # get the vector of a word, if exists

    if embedding_vector is not None:  # we found the word - add that words vector to the matrix
        return embedding_vector
    
    else:
         avg_vector = [-1.55611530e-01, -3.93543998e-03, -3.25425752e-02, -6.28335699e-02,
                          -4.78157075e-03, -1.84617653e-01, -4.94439378e-02, -1.80521324e-01,
                          -3.35793793e-02, -1.94202706e-01, -6.56424314e-02, 3.70132737e-02,
                          6.60830796e-01, -7.80616794e-03, -1.95153028e-04, 9.07416344e-02,
                          8.08839127e-02, 7.30239078e-02, 2.30256692e-01, 9.59603861e-02,
                          1.10939644e-01, 4.32463065e-02, 6.52063936e-02, -6.03170432e-02,
                          -2.05838501e-01, 7.50285745e-01, 1.29861072e-01, 1.11144960e-01,
                          -3.51636916e-01, 2.48395074e-02, 2.55199522e-01, 1.77181944e-01,
                          4.26653862e-01, -2.19325781e-01, -4.53459173e-01, -1.41409159e-01,
                          -8.41409061e-03, 1.01715224e-02, 6.76619932e-02, 1.28284395e-01,
                          6.85148776e-01, -1.77478697e-02, 1.28944024e-01, -7.42785260e-02,
                          -3.58294964e-01, 9.97241363e-02, -5.09560928e-02, 5.47902798e-03,
                          6.24788366e-02, 2.89107800e-01, 3.06909740e-01, 9.53846350e-02,
                          -8.97585154e-02, -3.03416885e-02, -1.80602595e-01, -1.63290858e-01,
                          1.23387389e-01, -1.73964277e-02, -6.13645613e-02, -1.17096789e-01,
                          1.49090782e-01, 1.17921308e-01, 1.05730975e-02, 1.33317500e-01,
                          -1.94899425e-01, 2.25606456e-01, 2.08363295e-01, 1.73583731e-01,
                          -4.40407135e-02, -6.87221363e-02, -1.83684096e-01, 7.04482123e-02,
                          -6.98078275e-02, 2.02260930e-02, 3.70468129e-03, 1.96141958e-01,
                          1.96837828e-01, 1.27971312e-02, 4.36565094e-02, 1.42354667e-01,
                          -3.62371027e-01, -1.10718250e-01, -4.84273471e-02, 4.64920104e-02,
                          -1.09924808e-01, -1.34851769e-01, 1.89310268e-01, -3.97192866e-01,
                          5.38146198e-02, -1.40333608e-01, 5.22745401e-02, 1.40163332e-01,
                          1.00092500e-01, 6.39176890e-02, 5.10458164e-02, 8.40307549e-02,
                          1.05783986e-02, 2.15598941e-01, -1.54302031e-01, 1.49716333e-01]
         return np.array(avg_vector, dtype='float64') 



def check_coverage(vocab, glove_model):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = glove_model[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


oov = check_coverage(tokenizer.word_counts, glove_model)
print('out of vocab: ', oov[:30])


embedding_dim = 100
num_tokens = len(word_index)
print('Vocabulary (number of unique words):', num_tokens)


# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = get_glove_vec(word)
    if embedding_vector is not None:
        
        embedding_matrix[i] = embedding_vector

print('embedding_matrix', embedding_matrix)
print('embedding_matrix %s.' % len(embedding_matrix))


savez_compressed('data\\embedding_matrix.npz', embedding_matrix)


# Convert Tag/Label to tag_index
count_KP = 0  
count_KP_words = 0 
count_NON_KP = 0  
y = [] 
for index, abstract in enumerate(tqdm(data['abstract'])):
    abstract_word_labels = [0] * len(abstract)  

    # add labels for words in abstract
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
        if not abstract_word_labels[i]:  # count NON-KPs
            count_NON_KP += 1
    count_KP_words += abstract_word_labels.count(1)  # count KP WORDS
    
    y.append(abstract_word_labels)

print('KP count: ', count_KP, '\nKP WORDS count: ', count_KP_words, '\nNON-KP count: ', count_NON_KP)


import json
# save data
with open(x_filename+".txt", "w") as fp_x:
    json.dump(X, fp_x)
with open(y_filename+".txt", "w") as fp_y:
    json.dump(y, fp_y)
    
    
print("Maximum length of title and abstract in the whole dataset", max_len)

for i in tqdm(range(0, len(X), batch_size)):

    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=0)
    if not x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed':
        y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)

y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]

filters = tables.Filters(complib='blosc', complevel=5)

# Save X batches into file
f = tables.open_file(x_filename+'.hdf', 'a')
ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters)
ds[:] = X_batch
f.close()

if not x_filename == 'data\\preprocessed_data\\x_TEST_data_preprocessed': 
    
    f = tables.open_file(y_filename + '.hdf', 'a')
    ds = f.create_carray('/', 'y_data' + str(i), obj=y_batch, filters=filters)
    ds[:] = y_batch
    f.close()
    

X_batch = None
y_batch = None


if y_filename == 'data\\preprocessed_data\\y_TEST_data_preprocessed':  # write ONLY for TEST DATA
    y_test = pd.DataFrame({'y_test_keyword': y})
    y_test['y_test_keyword'].to_csv(y_filename, index=False)  # save the preprocessed keyphrases
    
