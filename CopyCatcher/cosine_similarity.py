from __future__ import division
import pandas as pd
from gensim.models import doc2vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
import numpy
import csv

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

model = doc2vec.Doc2Vec(hashfxn=myhash)

#Load the model we trained earlier
model = doc2vec.Doc2Vec.load("../classifier/doc2vec/Doc2VecTaggedDocs")

sentence1 = preprocess("Why do rockets look white?")
sentence2 = preprocess("Why are rockets and boosters painted white?")

inferred_embedding_2 = numpy.array(model.infer_vector(doc_words=sentence2, steps=30, alpha=0.025))
inferred_embedding_1 = numpy.array(model.infer_vector(doc_words=sentence1, steps=30, alpha=0.025))

print(numpy.dot(inferred_embedding_1, inferred_embedding_2)/(numpy.linalg.norm(inferred_embedding_1,ord=2)*numpy.linalg.norm(inferred_embedding_2,ord=2)))

# Load the csv to proper format
test = pd.read_csv("../../train.csv", header=0)

output = []
for i in range(10000):
    sentence1 = preprocess(test.question1[i])
    sentence2 = preprocess(test.question2[i])

    inferred_embedding_1 = numpy.array(model.infer_vector(doc_words=sentence1, steps=100, alpha=0.025))
    inferred_embedding_2 = numpy.array(model.infer_vector(doc_words=sentence2, steps=100, alpha=0.025))

    cosine_similarity = numpy.dot(inferred_embedding_1, inferred_embedding_2)/(numpy.linalg.norm(inferred_embedding_1,ord=2)*numpy.linalg.norm(inferred_embedding_2,ord=2))
    output.append([cosine_similarity, test.is_duplicate[i]])

for i in range(1,51):
    accuracy = 0
    threshold = (i/50)
    for j in range(1,10000):
        if (output[j][0] <= threshold and output[j][1] == 0): 
            accuracy += (1/10000)
        if (output[j][0] > threshold and output[j][1] == 1): 
            accuracy += (1/10000)
    print(threshold, accuracy)

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(output)
