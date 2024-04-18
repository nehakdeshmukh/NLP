# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:18:49 2024

@author: nehak
"""

import time
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import tables
from tensorflow import constant  
from numpy import load
import Model, Input
from tensorflow.keras.layers import Embedding, LSTM,Bidirectional, Dense
from tf2crf import CRF

pd.set_option('display.max_columns', None)

start_time = time.time()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = ArgumentParser()

parser.add_argument("-sts", "--select_test_set", type=str,
                    help="select the test set to evaluate the model (options are:"
                         "'kp20k_full_abstract'"
                         "'nus_full_abstract'"
                         "'acm_full_abstract'"
                         "'semeval_full_abstract'"
                         ""
                         "'kp20k_sentences_abstract'"
                         "'nus_sentences_abstract'"
                         "'acm_sentences_abstract'"
                         "'semeval_sentences_abstract'"
                         ""
                         "'nus_sentences_fulltext'"
                         "'acm_sentences_fulltext'"
                         "'semeval_sentences_fulltext'"
                         ""
                         "'nus_paragraph_fulltext'"
                         "'acm_paragraph_fulltext'"
                         "'semeval_paragraph_fulltext'"
                         ""
                         "'nus_220_first_3_paragraphs'"
                         "'acm_220_first_3_paragraphs'"
                         "'semeval_220_first_3_paragraphs'"
                         "'nus_400_first_3_paragraphs'"
                         "'acm_400_first_3_paragraphs'"
                         "'semeval_400_first_3_paragraphs'"
                         ""
                         "'nus_summarization'"
                         "'acm_summarization'"
                         "'semeval_summarization'"
                         ")"
                    )

parser.add_argument("-pmp", "--pretrained_model_path", type=str,
                    help="the path and the name of the pretrained model")

parser.add_argument("-sm", "--sentence_model", type=int, default=0,
                    help="choose which data to load (options are: True for sentence model or False for whole title and abstracts model)")

args = parser.parse_args()


if args.sentence_model:
   
    # batch size, train and test data size
    batch_size = 256
    train_data_size = 4136306
    validation_data_size = 156519
    
    # Set input layer size
    MAX_LEN = 40 
    
    
else:
    # batch size,train and test data size
    batch_size = 64 
    train_data_size = 530390 
    validation_data_size = 20000 

    MAX_LEN = 400 
    
    
    
# embedding size, OUTPUT layer size
VECT_SIZE = 100  
number_labels = 2

doc_vocab = 321352

print('MAX_LEN of text', MAX_LEN)
print('VECT_SIZE', VECT_SIZE)
print('VOCABULARY', doc_vocab)


# Full abstract
if args.select_test_set=="kp20k_full_abstract":
    # [ test_data_size = 20000 ]
    test_data_size = 20000
    x_test_filename = 'data\\preprocessed_data\\data_train1\\x_TEST_data_preprocessed.hdf'  
    y_test_filename = 'data\\preprocessed_data\\data_train1\\y_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\data_train1\\x_TEST_preprocessed_TEXT'  
    y_filename = 'data\\preprocessed_data\\data_train1\\y_TEST_preprocessed_TEXT'  
    
elif args.select_test_set=="nus_full_abstract":
    # [ test_data_size = 211 ]
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_NUS_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_NUS_FULL_ABSTRACT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_full_abstract":
    # [ test_data_size = 2304 ]
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_TEST_vectors.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_TEST_vectors'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_ACM_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_ACM_FULL_ABSTRACT_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_full_abstract":
    # [ test_data_size = 244 ]
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\full_abstract\\x_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\full_abstract\\y_SEMEVAL_FULL_ABSTRACT_preprocessed_TEXT'

# Sentences abstract
elif args.select_test_set=="kp20k_sentences_abstract":
    # [ test_data_size = 155801 ]
    test_data_size = 155801
    x_test_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_data_preprocessed.hdf'  
    y_test_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\x_TEST_SENTENC_preprocessed_TEXT'  
    y_filename = 'data\\preprocessed_data\\y_TEST_SENTENC_preprocessed_TEXT'  
    
elif args.select_test_set=="nus_sentences_abstract":
    # [ test_data_size = 1673 ]
    test_data_size = 1673
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_NUS_SENTEC_ABSTRACT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_sentences_abstract":
    # [ test_data_size = 17481 ]
    test_data_size = 17481
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_ACM_SENTENC_ABSTRACT_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_sentences_abstract":
    # [ test_data_size = 1979 ]
    test_data_size = 1979
    x_test_filename = 'data\\preprocessed_data\\sentence_abstract\\x_SEMEVAL_SENTEC_ABSTRACT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_abstract\\y_SEMEVAL_SENTEC_ABSTRACT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_abstract\\x_SEMEVAL_SENTEC_ABSTRACT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_abstract\\y_SEMEVAL_SENTEC_ABSTRACT_preprocessed_TEXT'
    
# Sentences fulltext
elif args.select_test_set=="nus_sentences_fulltext":
    # [ test_data_size = 74219 ]
    test_data_size = 74219
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_NUS_SENTEC_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_sentences_fulltext":
    # [ test_data_size = 770263 ]
    test_data_size = 770263
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_ACM_SENTENC_FULLTEXT_preprocessed_TEXT'

elif args.select_test_set=="semeval_sentences_fulltext":
    # [ test_data_size = 75726 ]
    test_data_size = 75726
    x_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_SEMEVAL_SENTEC_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_SEMEVAL_SENTEC_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\sentence_fulltext\\x_SEMEVAL_SENTEC_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\sentence_fulltext\\y_SEMEVAL_SENTEC_FULLTEXT_preprocessed_TEXT'

# Paragraphs fulltext
elif args.select_test_set=="nus_paragraph_fulltext":
    # [ test_data_size = 4744 ]
    test_data_size = 4744
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_NUS_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_NUS_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_NUS_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_NUS_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_paragraph_fulltext":
    # [ test_data_size = 53083 ]
    test_data_size = 53083
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_ACM_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_paragraph_fulltext":
    # [ test_data_size = 5171 ]
    test_data_size = 5171
    x_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_SEMEVAL_PARAGRAPH_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_SEMEVAL_PARAGRAPH_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\paragraph_fulltext\\x_SEMEVAL_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\paragraph_fulltext\\y_SEMEVAL_PARAGRAPH_FULLTEXT_preprocessed_TEXT'
    
# First 3 paragraphs
elif args.select_test_set=="nus_220_first_3_paragraphs":
    MAX_LEN = 220
    test_data_size = 633
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="nus_400_first_3_paragraphs":
    test_data_size = 633
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_NUS_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_220_first_3_paragraphs":
    MAX_LEN = 220
    test_data_size = 6910
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="acm_400_first_3_paragraphs":
    test_data_size = 6910
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_ACM_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_220_first_3_paragraphs":
    MAX_LEN = 220
    test_data_size = 732
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\220\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_400_first_3_paragraphs":
    test_data_size = 732
    x_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\first_paragraphs_fulltext\\400\\y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT'
    
# Summarization of abstract and fulltext
elif args.select_test_set=="nus_summarization":
    test_data_size = 211
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_NUS_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    
elif args.select_test_set=="acm_summarization":
    test_data_size = 2304
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_ACM_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    
elif args.select_test_set=="semeval_summarization":
    test_data_size = 244
    x_test_filename = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed.hdf'
    y_test_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_TEST_data_preprocessed'
    x_filename = 'data\\preprocessed_data\\summarization_experiment\\x_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    y_filename = 'data\\preprocessed_data\\summarization_experiment\\y_SEMEVAL_FULLTEXT_SUMMARIZATION_preprocessed_TEXT'
    
else:
    print('WRONG ARGUMENTS! - please fill the argument "-sts" or "--select_test_set" with one of the proper values')
    sys.exit()

def load_data(x_filename, y_filename, batch_number):
 
    print('batch_number', batch_number)
    print(x_filename)

  
    with tables.File(x_filename, 'r') as h5f:
        x = h5f.get_node('/x_data' + str(batch_number)).read() 
       
        print('X SHAPE AFTER', np.array(x, dtype=object).shape)

    if not y_filename == '': 
        
        with tables.File(y_filename, 'r') as h5f:
            y = h5f.get_node('/y_data' + str(batch_number)).read() 
           
            print('y SHAPE AFTER', np.array(y, dtype=object).shape)

    if y_filename == '':  
        return x

    return x, constant(y)

def batch_generator(x_filename, y_filename, batch_size, number_of_batches):
    
    current_batch_number = 0 

    if 'TRAIN' in x_filename: 
        yield load_data(x_filename, y_filename, current_batch_number) 

    while True:
        yield load_data(x_filename, y_filename, current_batch_number)

       
        if current_batch_number < batch_size * number_of_batches:
            current_batch_number += batch_size
        else:
            current_batch_number = 0
            
# load dict of arrays
dict_data = load('data\\embedding_matrix.npz')
# extract the 1st array
embedding_matrix = dict_data['arr_0']
print(embedding_matrix)


from keras.regularizers import l1
from keras.constraints import max_norm
inpt = Input(shape=(None,)) 

model = Embedding(doc_vocab, output_dim=100, input_length=None,  
                  weights=[embedding_matrix], 
                  mask_zero=True, trainable=True, activity_regularizer=l1(0.0000001))(inpt) 

model = Bidirectional(LSTM(units=100, return_sequences=True, activity_regularizer=l1(0.0000000001), recurrent_constraint=max_norm(2)))(model) 

model = Dense(number_labels, activation=None)(model) 

crf = CRF()  
out = crf(model) 