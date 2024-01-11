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
from argparse import ArgumentParser
from pandas import json_normalize
import string

from tqdm import tqdm
tqdm.pandas()


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
