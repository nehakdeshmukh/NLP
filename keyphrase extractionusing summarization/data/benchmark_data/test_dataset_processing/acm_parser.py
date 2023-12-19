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

list_of_document_text = []  # save the text of documents
list_of_document_keyphrases = []  # save the keyphrases of documents
for key in keyphrases_dictionary:
	print('ACM/' + key + '.txt')
	print(keyphrases_dictionary[key])

	keyphrase_string = ''
	# format the keyphrases as key1;key2;key3 (required for preprocessing)
	for list_of_keyphrases in keyphrases_dictionary[key]:
		for keyphrase in list_of_keyphrases:
			keyphrase_string += keyphrase[0] + ';'
		list_of_document_keyphrases.append(keyphrase_string[:-1])  # [:-1] -> remove the ';' in the end

	# read the documents' text
	with open('ACM/' + key + '.txt', 'r', encoding="utf8") as document_text:
		list_of_document_text.append(document_text.read())