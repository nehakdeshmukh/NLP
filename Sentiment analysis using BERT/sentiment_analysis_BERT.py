# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:42:17 2023

@author: nehak
"""

import pandas as pd 
from sklearn.model_selection import train_test_split


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


