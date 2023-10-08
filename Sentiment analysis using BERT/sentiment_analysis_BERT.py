# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:42:17 2023

@author: nehak
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
from sklearn.metrics import f1_score

df = pd.read_csv('smile-annotations-final.csv', names=['id', 'text', 'category'])
df.set_index('id', inplace=True)

df.head()

df["category"].value_counts()


df = df[~df.category.str.contains('\|')]

df = df[df.category != 'nocode']


df.category.value_counts()

possible_labels = df.category.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
    
    
df['label'] = df.category.replace(label_dict)

# train test split 

x_train, x_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[x_train, 'data_type'] = 'train'
df.loc[x_val, 'data_type'] = 'val'


df.groupby(['category', 'label', 'data_type']).count()

#Tokenizer and Encoding

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_encoded_data = tokenizer.batch_encode_plus(df[df.data_type=='train'].text.values, 
                                                    add_special_tokens=True, 
                                                    return_attention_mask=True, 
                                                    pad_to_max_length=True, 
                                                    max_length=256, 
                                                    return_tensors='pt' )


val_encoded_data = tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values, 
                                                add_special_tokens=True, 
                                                return_attention_mask=True, 
                                                pad_to_max_length=True, 
                                                max_length=256, 
                                                return_tensors='pt')


input_ids_train = train_encoded_data['input_ids']
attention_masks_train = train_encoded_data['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = val_encoded_data['input_ids']
attention_masks_val = val_encoded_data['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

len(dataset_train)
len(dataset_val)

#BERT Pretrained model 

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)


dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')



