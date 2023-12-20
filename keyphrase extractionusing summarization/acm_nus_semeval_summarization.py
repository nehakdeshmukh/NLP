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

