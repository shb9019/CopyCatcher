#!/usr/bin/env python3

import pandas as pd
import nltk
import json
from bs4 import BeautifulSoup
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
from random import shuffle
import time
from sklearn.datasets import fetch_20newsgroups

#Choose tokenizer from nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def myhash(obj):
       return hash(obj) % (2 ** 32)

# Parse training data from training data CSV
train = pd.read_csv("../../train.csv", header=0)
num_questions = len(train.question1)

# # Here is the function review_to_sentences 
# def review_to_sentences(review, tokenizer, sentiment=""):
#     """
#     This function splits a review into parsed sentences
#     :param review:
#     :param tokenizer:
#     :param removeStopwords:
#     :return: sentences, list of lists
#     """
#     # review.strip()remove the white spaces in the review
#     # use tokenizer to separate review to sentences

#     n = len(review)
#     temp = ""
#     for i in range(n):
#         if review[i] == '"':
#             temp += ' '
#         else:
#             temp += review[i]
#     review = temp
#     rawSentences = tokenizer.tokenize(review.strip())
#     cleanedReview = []
#     for sentence in rawSentences:
#         if len(sentence) > 0:
#             sentence = re.sub("[^a-zA-Z]", " ", sentence)
#             cleanedReview += sentence

#     # if(sentiment != ""):
#     #     cleanedReview.append(sentiment)

#     return cleanedReview

# labeled = []
# labelized = []

# index = 0
# for i in range(num_questions):
#     try:
#         labeled.append(review_to_sentences(train.question1[i], tokenizer, train.is_duplicate[i]))
#     except:
#         continue

#     tag_class = "0"
    
#     if train.is_duplicate[i] == 1:
#         tag_class = "1"
    
#     labeled[index].append(tag_class)
#     labelized.append(TaggedDocument(words=labeled[index], tags=['%s_%s'%('LABELED', index),tag_class]))
#     index += 1

# model = doc2vec.Doc2Vec(vector_size=300,
#                         window=10, min_count=40,
#                         sample=1e-3, workers=4,hashfxn=myhash)

# model.build_vocab(labelized)

# #Train the model for 10 epochs
# for epoch in range(1,10):
    
#     print("Starting Epoch ",epoch)
    
    
#     start_time = time.time()
#     #Shuffle the tagged cleaned up reviews in each epoch
#     shuffle(labelized)

#     model.train(labelized, total_examples=model.corpus_count, epochs=1)
    
#     print("Epoch ",epoch," took %s minutes " % ((time.time() - start_time)/60))

# #Save the trained model
# model.save("../classifier/doc2vec/Doc2VecTaggedDocs")
