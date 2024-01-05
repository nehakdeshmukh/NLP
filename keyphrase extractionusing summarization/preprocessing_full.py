# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:53:09 2024

@author: nehak
"""

#preprocessing_full

import re
import sys
import json
import pickle
import pandas as pd
from argparse import ArgumentParser


from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)

parser = ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="choose which type of data to create (options are: train, validation or test)")

args = parser.parse_args()