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


list_of_document_title = []  # save the title of documents
list_of_document_abstract = []  # save the abstract of documents
list_of_document_text = []  # save the body of documents
list_of_document_keyphrases = []  # save the keyphrases of documents
for key in keyphrases_dictionary:

    keyphrase_string = ''
    
    for list_of_keyphrases in keyphrases_dictionary[key]:
        for keyphrase in list_of_keyphrases:
            for nested_kp in keyphrase: 
                keyphrase_string += nested_kp + ';'
        list_of_document_keyphrases.append(keyphrase_string[:-1])  



parser = etree.XMLParser()  
path = 'semeval_2010/train_test_combined/' + key + '.xml'

sentences = []  
tree = etree.parse(path, parser)