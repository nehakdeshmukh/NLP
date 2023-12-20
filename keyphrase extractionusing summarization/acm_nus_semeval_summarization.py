# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:48:59 2023

@author: nehak
"""

import time
import json
from datetime import timedelta
from pandas import json_normalize
from extractive import ExtractiveSummarizer

model = ExtractiveSummarizer.load_from_checkpoint("models\\epoch=3.ckpt")

file = 'datasets\\ACM.json' 

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)