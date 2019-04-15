#!/usr/bin/env python3

import pandas as pd
import nltk
import json
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
from random import shuffle
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
import xml.etree.ElementTree as ET
root = ET.parse('../Posts.xml').getroot()

nltk.download('wordnet')
nltk.download('stopwords')

en_stops = set(stopwords.words('english'))

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def myhash(obj):
       return hash(obj) % (2 ** 32)

data = []

for row in root:
    if 'Tags' in row.attrib.keys():
        data.append([preprocess(row.attrib['Title'] + ' ' + row.attrib['Body']), preprocess(row.attrib['Tags'])])

taggedData = []

for question in data:
    taggedData.append(TaggedDocument(words=question[0], tags=question[1]))

model = doc2vec.Doc2Vec(vector_size=300,
                        window=10, min_count=10,
                        sample=1e-3, workers=4,hashfxn=myhash)

model.build_vocab(taggedData)

#Train the model for 10 epochs
for epoch in range(1,11):
    
    print("Starting Epoch ",epoch)
    
    start_time = time.time()
    #Shuffle the tagged cleaned up reviews in each epoch
    shuffle(taggedData)

    model.train(taggedData, total_examples=model.corpus_count, epochs=1)
    
    print("Epoch ",epoch," took %s minutes " % ((time.time() - start_time)/60))

#Save the trained model
model.save("../classifier/doc2vec/Doc2VecTaggedDocs")
