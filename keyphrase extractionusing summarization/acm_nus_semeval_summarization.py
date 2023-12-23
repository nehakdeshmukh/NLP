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

from tqdm import tqdm
tqdm.pandas()

model = ExtractiveSummarizer.load_from_checkpoint("models\\epoch=3.ckpt")

file = 'datasets\\ACM.json' 

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))

# convert json to dataframe
data = json_normalize(json_data)

print(data)

def extract_title(fulltext):
    # extract the title
    start_title = fulltext.find("--T\n") + len("--T\n")  # skip the special characters '--T\n'
    end_title = fulltext.find("--A\n")
    title = fulltext[start_title:end_title]

    return title

data['title'] = data['fulltext'].apply(extract_title)

start_time = time.time()

for index, fulltext in enumerate(tqdm(data['fulltext'])):
    # extract the abstract
    start_abstract = fulltext.find("--A\n") + len("--A\n")  # skip the special characters '--A\n'
    end_abstract = fulltext.find("--B\n")
    abstract = fulltext[start_abstract:end_abstract]

    start_fulltext = fulltext.find("--B\n") + len("--B\n")  # skip the special characters '--B\n'
    end_fulltext = fulltext.find("--R\n")  # do not include references
    main_body = fulltext[start_fulltext:end_fulltext]

    abstract_mainBody = abstract + ' ' + main_body
    
    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['fulltext'].iat[index] = summarize_fulltext
    
data.rename(columns={"fulltext": "abstract"}, inplace=True)

print(data)
print(data['abstract'][0])
print(data['abstract'][50])

total_time = str(timedelta(seconds=(time.time() - start_time)))
print("\n--- ACM %s running time ---" % total_time)


summarized_file = 'datasets\\summarized_text\\ACM_summarized.csv'  # TEST data to evaluate the final model

data[['title', 'abstract', 'keyword']].to_csv(summarized_file, index=False)

file = 'datasets\\semeval_2010.json'

json_data = []
for line in open(file, 'r', encoding="utf8"):
    json_data.append(json.loads(line))


data = json_normalize(json_data)

print(data)

start_time = time.time()

for index, abstract in enumerate(tqdm(data['abstract'])):
    
    abstract_mainBody = abstract + ' ' + data['fulltext'][index]

    abstract_mainBody = abstract_mainBody.replace('\n', ' ')

    summarize_fulltext = model.predict(abstract_mainBody, num_summary_sentences=14)

    data['abstract'].iat[index] = summarize_fulltext





