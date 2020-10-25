import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import numpy as np
import pandas as pd
import nltk
from collections import Counter
import pickle
from utils import getAvgFeatureVecs, featureVecMethod, tokenize_data

"""
EMBEDDING MODEL
Google News
3 million word vectors
300 dimensions

EMOTION CLASSIFIER
Support Vector Machine
With stopwords
Frequency threshold = 1
2000 iterations
"""

# Word frequency = 1
frequency_threshold = 1

# Include stopwords
stop_words = []

# Load embedding model
filename = 'emb_models/GoogleNews-vectors-negative300.bin'
word_embedding_model = KeyedVectors.load_word2vec_format(filename, binary=True)

# This model has 300 dimensions so we set the number of features to 300
num_features = 300

# Transform training data to use
filepath = 'data/MELD/train_sent_emo.csv'
dftrain = pd.read_csv(filepath)

dftrain['Utterance'] = dftrain['Utterance'].str.replace("\x92|\x97|\x91|\x93|\x94|\x85", "'")

filepath = './data/MELD/test_sent_emo.csv'
dftest = pd.read_csv(filepath)
dftest['Utterance'] = dftest['Utterance'].str.replace("\x92|\x97|\x91|\x93|\x94|\x85", "'")

training_instances = tokenize_data(dftrain['Utterance'])
training_labels = tokenize_data(dftrain['Emotion'])

test_instances = tokenize_data(dftest['Utterance'])
test_labels = tokenize_data(dftest['Emotion'])

frequent_keywords = []
alltokens = []
for utterance in dftrain['Utterance']:
    tokenlist = nltk.tokenize.word_tokenize(utterance)
    for token in tokenlist:
        alltokens.append(token)

kw_counter = Counter(alltokens)

for word, count in kw_counter.items():
    if count>frequency_threshold:
        frequent_keywords.append(word)

unknown_words =[]
known_words = []

index2word_set = set(word_embedding_model.index2word)

# Vectorize
trainDataVecs = getAvgFeatureVecs(training_instances, frequent_keywords, stop_words, word_embedding_model, index2word_set, num_features)

testDataVecs = getAvgFeatureVecs(test_instances, frequent_keywords, stop_words, word_embedding_model, index2word_set, num_features)

# Training the classifier
label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(training_labels+test_labels)

training_classes = label_encoder.transform(training_labels)
test_classes = label_encoder.transform(test_labels)

svm_linear_clf = svm.LinearSVC(max_iter=2000)
svm_linear_clf.fit(trainDataVecs, training_classes)

# Saving the classifier
filename_classifier = 'emo_models/svm_stopwords_gn.sav'
pickle.dump(svm_linear_clf, open(filename_classifier, 'wb'))

# Saving the frequent keywords and label encoder
filename_freq_keywords = 'emo_models/stopwords_frequent_keywords.sav'
pickle.dump(frequent_keywords, open(filename_freq_keywords, 'wb'))

filename_encoder = 'emo_models/gn_label_encoder.sav'
pickle.dump(label_encoder, open(filename_encoder, 'wb'))
