# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 07:55:24 2024

@author: nehak
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser


from tqdm import tqdm
tqdm.pandas()


pd.set_option('display.max_columns', None)

parser = ArgumentParser()

parser.add_argument("-m", "--mode", type=str,
                    help="choose which type of data to create (options are: train, validation or test)")