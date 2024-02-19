# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:40:19 2024

@author: nehak
"""

import pandas as pd
import time
from argparse import ArgumentParser




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

parser.add_argument("-sm", "--sentence_model", type=int, default=0,
                    help="choose which data to load (options are: True for sentence model or False for whole title and abstracts model)")

args = parser.parse_args()


if args.sentence_model:
   
    batch_size = 256
    train_data_size = 4136306
    validation_data_size = 156519
    MAX_LEN = 40
else:
    
    batch_size = 64
    train_data_size = 530390 
    validation_data_size = 20000 
    MAX_LEN = 400 

# Set embedding size, OUTPUT layer size
VECT_SIZE = 100  
number_labels = 2 
doc_vocab = 321352


print('MAX_LEN of text', MAX_LEN)
print('VECT_SIZE', VECT_SIZE)
print('VOCABULARY', doc_vocab)