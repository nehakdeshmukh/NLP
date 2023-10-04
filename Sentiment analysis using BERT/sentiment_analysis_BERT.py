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
