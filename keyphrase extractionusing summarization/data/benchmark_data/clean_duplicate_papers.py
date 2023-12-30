# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 07:18:01 2023

@author: nehak
"""

import json
import string
from pandas import json_normalize

from tqdm import tqdm
tqdm.pandas()


file_kp20k = '..\\kp20k_training.json'  # TRAIN data
file_kp20k_val = '..\\kp20k_validation.json'  # VALIDATION data
file_kp20k_test = '..\\kp20k_testing.json'  # TEST data 

file_nus = 'NUS.json'  # TEST data 
file_acm = 'ACM.json'  # TEST data 
file_semeval = 'semeval_2010.json'  # TEST data 

punctuation = string.punctuation + '\t' + '\n'
table = str.maketrans(punctuation, ' '*len(punctuation))  # OR {key: None for key in string.punctuation}

print(punctuation, 'LEN:', len(punctuation))


def remove_punct(text):
    clean_text = text.translate(table)
    return clean_text


json_data = []
for line in open(file_kp20k, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_kp20k = json_normalize(json_data)

# print(data_kp20k)

# remove punctuation
data_kp20k['clean_title'] = data_kp20k['title'].apply(remove_punct)
# remove whitespaces
data_kp20k['clean_title'] = data_kp20k["clean_title"].str.replace('\s+', ' ', regex=True)

# KP Validation

json_data = []
for line in open(file_kp20k_val, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# json to dataframe
data_kp20k_val = json_normalize(json_data)

# print(data_kp20k_val)

# remove punctuation
data_kp20k_val['title'] = data_kp20k_val['title'].apply(remove_punct)
# remove whitespaces
data_kp20k_val['title'] = data_kp20k_val["title"].str.replace('\s+', ' ', regex=True)



# KP test
json_data = []
for line in open(file_kp20k_test, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

#  json to dataframe
data_kp20k_test = json_normalize(json_data)

data_kp20k_test['title'] = data_kp20k_test['title'].apply(remove_punct)

data_kp20k_test['title'] = data_kp20k_test["title"].str.replace('\s+', ' ', regex=True)


json_data = []
for line in open(file_nus, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# json to dataframe
data_nus = json_normalize(json_data)

print(data_nus)

data_nus['title'] = data_nus['title'].apply(remove_punct)
# remove whitespaces
data_nus['title'] = data_nus["title"].str.replace('\s+', ' ', regex=True)


json_data = []
for line in open(file_acm, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_acm = json_normalize(json_data)

for index, fulltext in enumerate(data_acm['fulltext']):
    # extract the title
    start_title = fulltext.find("--T\n") + len("--T\n")  #special characters '--T\n'
    end_title = fulltext.find("--A\n")
    title = fulltext[start_title:end_title]
    title = title.translate(table)  # remove punctuation
    data_acm['fulltext'].iat[index] = title

data_acm.rename(columns={"fulltext": "title"}, inplace=True)

# remove whitespaces
data_acm['title'] = data_acm["title"].str.replace('\s+', ' ', regex=True)


# # Load SemEval

json_data = []
for line in open(file_semeval, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data_semeval = json_normalize(json_data)

# remove punctuation
data_semeval['title'] = data_semeval['title'].apply(remove_punct)
# remove whitespaces
data_semeval['title'] = data_semeval["title"].str.replace('\s+', ' ', regex=True)

