# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:33:14 2023

@author: nehak
"""

import json
import pandas as pd
from pandas import json_normalize
import xml.etree.ElementTree as etree


keyphrases_file = 'semeval_2010/train_test.combined.stem.json'


with open(keyphrases_file, 'r', encoding="utf8") as json_file:
    json_data = json.load(json_file)  # loads
print(json_data)

# convert json to dataframe
keyphrases_dictionary = json_normalize(json_data)

print(keyphrases_dictionary)


