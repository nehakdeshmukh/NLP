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

for sentence in tree.iterfind('./document/sentences/sentence'):
    
    starts = [int(u.text) for u in
              sentence.iterfind("tokens/token/CharacterOffsetBegin")]
    ends = [int(u.text) for u in
            sentence.iterfind("tokens/token/CharacterOffsetEnd")]
    sentences.append({
        "words": [u.text for u in
                  sentence.iterfind("tokens/token/word")],
        "lemmas": [u.text for u in
                   sentence.iterfind("tokens/token/lemma")],
        "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
        "char_offsets": [(starts[k], ends[k]) for k in
                         range(len(starts))]
    })
    sentences[-1].update(sentence.attrib)
    
title = ''
abstract = ''
body = ''
for indx, sent in enumerate(sentences):
    if sentences[indx]['section'] == 'title':  # for the title
        title += ' ' + ' '.join(sentences[indx]['words'])
    elif sentences[indx]['section'] == 'abstract':  # for the abstract
        abstract += ' ' + ' '.join(sentences[indx]['words'])
    else:  # for the main body (everything else)
        body += ' ' + ' '.join(sentences[indx]['words'])

list_of_document_title.append(title)
list_of_document_abstract.append(abstract)
list_of_document_text.append(body)

