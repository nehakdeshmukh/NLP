# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 07:55:24 2024

@author: nehak
"""
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser


from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)

parser = ArgumentParser()

parser.add_argument("-m", "--mode", type=str,
                    help="choose which type of data to create (options are: train, validation or test)")

parser.add_argument("-sm", "--sentence_model", type=bool, default=False,
                    help="choose which data to load (options are: True for data split in sentences or False for whole title and abstracts)")

args = parser.parse_args()


if args.sentence_model:
  
    batch_size = 352
    max_len = 40  

    if args.mode == 'train':
        # Define the file paths for TRAIN data
        x_filename = 'data\\preprocessed_data\\x_TRAIN_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TRAIN_SENTENC_data_preprocessed'
        
    elif args.mode == 'validation':
        # Define the file paths for VALIDATION data 
        x_filename = 'data\\preprocessed_data\\x_VALIDATION_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_VALIDATION_SENTENC_data_preprocessed'
        
    elif args.mode == 'test':
        # Define the file paths for TEST data 
        x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed'
        
    else:
        print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
        sys.exit()
