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