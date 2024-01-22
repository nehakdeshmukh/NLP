# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:21:06 2024

@author: nehak
"""
import re
import sys
import json
import pickle
import string
import tables
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)


parser = ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="choose which type of data to create (options are: train, validation & test)")

args = parser.parse_args()