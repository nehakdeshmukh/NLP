# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:00:56 2023

@author: nehak
"""

import json
import pandas as pd
from pandas import json_normalize


keyphrases_file = 'ACM/all_keys_in_json/test.author.stem.json'


with open(keyphrases_file, 'r', encoding="utf8") as json_file:
    json_data = json.load(json_file)  # loads
print(json_data)

# convert json to dataframe
keyphrases_dictionary = json_normalize(json_data)

print(keyphrases_dictionary)