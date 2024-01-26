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
