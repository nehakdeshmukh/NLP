# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:18:37 2024

@author: nehak
"""
import json
import pandas as pd
from pandas import json_normalize

file = 'data\\kp20k_validation.json' 



# VALIDATION data path
x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'



#  TEST data - use for EVALUATION (exact/partial matching)
x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  
y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  


batch_size = 32  

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)