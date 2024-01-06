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
    # reading the initial JSON data using json.load()
    file = 'data\\kp20k_validation.json'  # VALIDATION data to tune model parameters

    # Define the file paths and names to save VALIDATION data to tune model parameters
    x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'
elif args.mode == 'test':
    
    file = 'data\\kp20k_testing.json'  
    x_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
    y_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'

else:
    print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
    sys.exit()

    x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT' 
    y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT' 


batch_size = 64  # 1024  # 10000
max_len = 400 


json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)