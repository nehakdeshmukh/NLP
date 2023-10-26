# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:12:19 2023

@author: nehak
"""

import pandas as pd
import numpy as np

import os
import sys

Ravdess  = "Ravdess/audio_speech_actors_01-24/"
Savee = "ALL/"


ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []


for dir in ravdess_directory_list:
    
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        