#!/usr/bin/env python3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from nltk.metrics import edit_distance
from sklearn.metrics import log_loss
import re
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def tokenize(q1, q2):
    """
        q1 and q2 are sentences/questions. Function returns a list of tokens for both.
        "hi how are you" => ["hi", "how", "are", "you"]
    """
    return word_tokenize(q1), word_tokenize(q2)


def posTag(q1, q2):
    """
        q1 and q2 are lists. Function returns a list of POS tagged tokens for both.
        ["hi", "there"] => [('hi', 'NN'), ('there', 'EX')]
    """
    return nltk.pos_tag(q1), nltk.pos_tag(q2)


def stemmer(tag_q1, tag_q2):
    """
        tag_q = tagged lists. Function returns a stemmed list.
    """
    stem_q1 = []
    stem_q2 = []

    for token in tag_q1:
        stem_q1.append(stem(token))

    for token in tag_q2:
        stem_q2.append(stem(token))

    return stem_q1, stem_q2

class Lesk(object):

    def __init__(self, sentence):
        """Given a string sentence, break sentence into words"""
        self.sentence = sentence
        self.meanings = {}
        for word in sentence:
            self.meanings[word] = ''

    def getSenses(self, word):
        """Get all synsets of a word
        Synsets is the basic unit in wordnet

        Example:

        getSenses(bank) = [Synset('bank.n.01'), Synset('depository_financial_institution.n.01'), Synset('bank.n.03'), 
        Synset('bank.n.04'), Synset('bank.n.05'), Synset('bank.n.06'), Synset('bank.n.07'), Synset('savings_bank.n.02'), 
        Synset('bank.n.09'), Synset('bank.n.10'), Synset('bank.v.01'), Synset('bank.v.02'), Synset('bank.v.03'), 
        Synset('bank.v.04'), Synset('bank.v.05'), Synset('deposit.v.02'), Synset('bank.v.07'), Synset('trust.v.01')]
        """
        return wn.synsets(word.lower())

    def getGloss(self, senses):
        """Tokenizes definition for each synset and returns. Used for WSD
        
        Example:

        getGloss(getSenses("bank")) = {
            'bank.n.01': ['sloping', 'land', '(', 'especially', 'the', 'slope', 'beside', 'a', 'body', 
            'of', 'water', ')'], 'depository_financial_institution.n.01': ['a', 'financial', 'institution', 'that', 'accepts', 
            'deposits', 'and', 'channels', 'the', 'money', 'into', 'lending', 'activities'], 'bank.n.03': ['a', 'long', 'ridge', 
            'or', 'pile'], 'bank.n.04': ['an', 'arrangement', 'of', 'similar', 'objects', 'in', 'a', 'row', 'or', 'in', 'tiers'], 
            'bank.n.05': ['a', 'supply', 'or', 'stock', 'held', 'in', 'reserve', 'for', 'future', 'use', '(', 'especially', 'in', 
            'emergencies', ')'], 'bank.n.06': ['the', 'funds', 'held', 'by', 'a', 'gambling', 'house', 'or', 'the', 'dealer', 'in', 
            'some', 'gambling', 'games'], 'bank.n.07': ['a', 'slope', 'in', 'the', 'turn', 'of', 'a', 'road', 'or', 'track', ';', 
            'the', 'outside', 'is', 'higher', 'than', 'the', 'inside', 'in', 'order', 'to', 'reduce', 'the', 'effects', 'of', 
            'centrifugal', 'force'], 'savings_bank.n.02': ['a', 'container', '(', 'usually', 'with', 'a', 'slot', 'in', 'the', 
            'top', ')', 'for', 'keeping', 'money', 'at', 'home'], 'bank.n.09': ['a', 'building', 'in', 'which', 'the', 'business', 
            'of', 'banking', 'transacted'], 'bank.n.10': ['a', 'flight', 'maneuver', ';', 'aircraft', 'tips', 'laterally', 'about', 
            'its', 'longitudinal', 'axis', '(', 'especially', 'in', 'turning', ')'], 'bank.v.01': ['tip', 'laterally'], 'bank.v.02': 
            ['enclose', 'with', 'a', 'bank'], 'bank.v.03': ['do', 'business', 'with', 'a', 'bank', 'or', 'keep', 'an', 'account', 'at',
            'a', 'bank'], 'bank.v.04': ['act', 'as', 'the', 'banker', 'in', 'a', 'game', 'or', 'in', 'gambling'], 'bank.v.05': ['be',
            'in', 'the', 'banking', 'business'], 'deposit.v.02': ['put', 'into', 'a', 'bank', 'account'], 'bank.v.07': ['cover',
            'with', 'ashes', 'so', 'to', 'control', 'the', 'rate', 'of', 'burning'], 'trust.v.01': ['have', 'confidence', 'or',
            'faith', 'in']
        }

        """
        gloss = {}
        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())
        return gloss

    def getAll(self, word):
        """
        1. get all synsets of a word
        2. tokenize each synset definition as a dict
        3. return {synset.name => [tokenized definition]}
        """
        senses = self.getSenses(word)

        if senses == []:
            return {word.lower(): senses}

        return self.getGloss(senses)

    def Score(self, set1, set2):
        """Returns overlap between two sets. """
        # Base
        overlap = 0

        # Step
        for word in set1:
            if word in set2:
                overlap += 1

        return overlap

    def overlapScore(self, word1, word2):
        """
        1. Get dict for word1 and word2
        2. For each tokenized word (which is a synset of word1) in dict of word1, see if its in dict2
        3. Calculate score based on this
        4. Calculate synset based on highest score
        """
        gloss_set1 = self.getAll(word1)
        # if the meaning of the the second word has already been determined before, use gloass of that word
        # otherwise find all possible meanings to find similarity
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        score = {}
        for i in gloss_set1.keys():
            # for each word in the first word's set (synset), use its definition against all other definitions of the other
            # words synsets and get score. The one with the max score is picked as the correct word sense.
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        bestSense = None
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        """
        Given a word and a sentence, gives word and meaning of the word in context to the sentence. Uses overlap as metric

        Returns word, synset.name, meaning

        lazy faineant.s.01 disinclined to work or exertion
        """
        maxOverlap = 0
        context = sentence
        word_sense = []
        meaning = {}
        # get all synsets of the word
        senses = self.getSenses(word)

        for sense in senses:
            # for each meaning of the word, assign score 0 initially
            meaning[sense.name()] = 0

        # for each word in the context, get overlapscore
        # here context = sentence in which the word appears
        for word_context in context:
            # here, we're using the context provided by the other words to see which is the correct
            # gloss (meaning) of the word we're trying to get
            if not word == word_context:
                score = self.overlapScore(word, word_context)
                if score[0] == None:
                    continue
                meaning[score[0]] += score[1]

        if senses == []:
            return word, None, None

        self.meanings[word] = max(meaning.keys(), key=lambda x: meaning[x])

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()

def path(set1, set2):
    """Returns shortest number of edges between those two word sensesbetween two synsets s1 and s2
    """
    return wn.path_similarity(set1, set2)

def wup(set1, set2):
    """
    LCH = -log(path)
    WUP = very similar to LCH, except it weights the edges based on distance in the hierarchy. 
    For example, jumping from inanimate to animate is a larger distance than jumping from say Felid to Canid
    ie, the edges are weighed such that changes in type cost more.
    """
    return wn.wup_similarity(set1, set2)

def edit(word1, word2):
    if float(edit_distance(word1, word2)) == 0.0:
        return 0.0
    return 1.0 / float(edit_distance(word1, word2))

def computePath(q1, q2):
    """ Takes two meaning sentences.
    [(name, synset.name, meaning)]
    """
    R = np.zeros((len(q1), len(q2)))
    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                # if either words meaning was not found in synset, fall back to levenshtein distance of the two words
                sim = edit(q1[i][0], q2[j][0])
            else:
                # else convert to synset and use path similarity. that is, how far away they are in the tree struct of wordnet
                sim = path(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            # if the similarity above turns out to be None (happens when wordnet isnt updated about the relationship b/w words)
            # fall back to levenshtein distance
            if sim == None:
                sim = edit(q1[i][0], q2[j][0])
            # update distance between words in R matrix and return R
            R[i, j] = sim
    return R

def computeWup(q1, q2):
    """Similar, except uses WUP as scoring factor
    """
    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = wup(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim
    return R

def overallSim(q1, q2, R):
    """
    This formula is proposed by Wu & Palmer, the measure takes into account both path
    length and depth of the least common sub-summer :
        Sim(s, t) = 2 * depth(LCS)/[depth(s) + depth(t)]
    """

    sum_X = 0.0
    sum_Y = 0.0

    for i in range(len(q1)):
        max_i = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_i:
                max_i = R[i, j]
        sum_X += max_i

    # trying to modify formula to get distances both ways
    for i in range(len(q2)):
        max_j = 0.0
        for j in range(len(q1)):
            if R[j, i] > max_j:
                max_j = R[j, i]
        sum_Y += max_j
        
    if (float(len(q1)) + float(len(q2))) == 0.0:
        return 0.0
        
    overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))
    return overall

def semanticSimilarity(q1, q2):

    tokens_q1, tokens_q2 = tokenize(q1, q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    sentence = []
    # check similarity only using the main words in the sentence.
    for i, word in enumerate(tag_q1):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense1 = Lesk(sentence)
    sentence1Means = []
    for word in sentence:
        # append most suitable meaning / synsets name of each word given sentence
        sentence1Means.append(sense1.lesk(word, sentence))

    # similarly do for sentence2
    sentence = []
    for i, word in enumerate(tag_q2):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense2 = Lesk(sentence)
    sentence2Means = []
    for word in sentence:
        sentence2Means.append(sense2.lesk(word, sentence))

    # sentencexMeans is an array of form [(word, synset.name, meaning)]
    R1 = computePath(sentence1Means, sentence2Means)
    R2 = computeWup(sentence1Means, sentence2Means)

    # take average of the two scores
    R = (R1 + R2) / 2


    return overallSim(sentence1Means, sentence2Means, R)

STOP_WORDS = nltk.corpus.stopwords.words()
def clean_sentence(val):
    """Remove chars that are not letters or numbers, downcase, then remove stop words"""
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence

X_train = pd.read_csv('../input/quora.csv')
X_train = X_train.dropna(how="any")
X_train = X_train.sample(n=100)

y = X_train['is_duplicate']

print('Cleaning data')
for col in ['question1', 'question2']:
    X_train[col] = X_train[col].apply(clean_sentence)

y_pred = []
count = 0
print('Semantic Similarity being calculated')
for row in X_train.itertuples():
    # print row
    q1 = str(row[4])
    q2 = str(row[5])

    sim = semanticSimilarity(q1, q2)
    count += 1
    if count % 10000 == 0:
        print(str(count)+", "+str(sim)+", "+str(row[6]))
    y_pred.append(sim)

print("Accuracy = {} %".format(100*log_loss(y, np.array(y_pred))))

