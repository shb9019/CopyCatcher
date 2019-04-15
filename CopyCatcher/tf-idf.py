from __future__ import division
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
from scipy import spatial
from string import Template

train_set = []  # Documents
test_set = []  # Query
stopWords = stopwords.words('english')

test = pd.read_csv("../../train.csv", header=0)

print("Using the quora data set...")
print("Total data set size" , len(test.question1))
print("Training data set size" , len(test.question1)//18)

for i in range(len(test.question1)//18):
    train_set.append(test.question1[i])
    train_set.append(test.question2[i])

vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()

print("Count vector is built")

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
transformer.fit(trainVectorizerArray)

print("Tf-Idf vector is learnt from count vector")

output = []
test_set = []
for i in range(10000):
    test_set = []
    test_set.append(test.question1[i])
    test_set.append(test.question2[i])
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    tfidf = transformer.transform(testVectorizerArray)
    similarilty = 1 - spatial.distance.cosine((tfidf.todense())[0][0], (tfidf.todense())[1][0])
    output.append([similarilty, test.is_duplicate[i]])

for i in range(1,51):
    accuracy = 0
    threshold = (i/50)
    for j in range(1,10000):
        if (output[j][0] <= threshold and output[j][1] == 0): 
            accuracy += (1/10000)
        if (output[j][0] > threshold and output[j][1] == 1): 
            accuracy += (1/10000)
    print(threshold, accuracy*100)
