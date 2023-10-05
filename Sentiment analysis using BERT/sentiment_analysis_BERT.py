# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:42:17 2023

@author: nehak
"""

import pandas as pd 

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
    
    


