# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 07:55:24 2024

@author: nehak
"""
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tables


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
        
        
else:
    batch_size = 64 
  
    max_len = 400  
    
    if args.mode == 'train':
        # TRAIN data
        x_filename = 'data\\preprocessed_data\\x_TRAIN_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TRAIN_data_preprocessed'
        
    elif args.mode == 'validation':
        # VALIDATION data 
        x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'
        
    elif args.mode == 'test':
        # TEST data 
        x_filename = 'data\\preprocessed_data\\x_TEST_data_preprocessed'
        y_filename = 'data\\preprocessed_data\\y_TEST_data_preprocessed'
    else:
        print('WRONG ARGUMENTS! - please fill the argument "-m" or "--mode" with one of the values "train", "validation" or "test"')
        sys.exit()

print("Βatch size", batch_size)
print("Maximum length of title and abstract in the whole dataset", max_len)


import json
with open(x_filename+".txt", "r") as fp:
    X = json.load(fp)
with open(y_filename+".txt", "r") as fp:
    y = json.load(fp)

print('X SHAPE', pd.DataFrame(X).shape)  


for i in tqdm(range(0, len(X), batch_size)):

    X_batch = pad_sequences(sequences=X[i:i + batch_size], padding="post", maxlen=max_len, value=0)
    
    if 'TEST' not in x_filename:
        y_batch = pad_sequences(sequences=y[i:i + batch_size], padding="post", maxlen=max_len, value=0)
    
        y_batch = [to_categorical(i, num_classes=2, dtype='int8') for i in y_batch]

    
    filters = tables.Filters(complib='blosc', complevel=5)

    
    f = tables.open_file(x_filename+'.hdf', 'a')
    ds = f.create_carray('/', 'x_data'+str(i), obj=X_batch, filters=filters)
    ds[:] = X_batch
    #print(ds)
    f.close()
    
    
    if 'TEST' not in x_filename:  # do NOT write for TEST DATA
        f = tables.open_file(y_filename + '.hdf', 'a')
        ds = f.create_carray('/', 'y_data' + str(i), obj=y_batch, filters=filters)
        ds[:] = y_batch
        f.close()
    
    
    X_batch = None
    y_batch = None
    
    with tables.File(x_filename+'.hdf', 'r') as h5f:
        x = h5f.get_node('/x_data'+str(1024)).read()
        print(x)
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)
    