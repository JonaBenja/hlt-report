import gensim
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import sklearn
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pickle
from utils import getAvgFeatureVecs, featureVecMethod, tokenize_data

"""
EMBEDDING MODEL
GloVe embeddings
27B word vectors
200 dimensions

EMOTION CLASSIFIER
Support Vector Machine
Stopwords excluded
Frequency threshold = 10
2000 iterations
"""

# Word frequency = 10
frequency_threshold = 10

# Exclude stopwords
stop_words = set(stopwords.words('english'))

# Load embedding model
glove_file = datapath('/Users/jonabenja/Desktop/glove.twitter.27B/glove.twitter.27B.200d.txt')
tmp_file = get_tmpfile('test_word2vec.txt')

wordembeddings = glove2word2vec(glove_file, tmp_file)
word_embedding_model = KeyedVectors.load_word2vec_format(tmp_file)

# This model has 200 dimensions so we set the number of features to 200
num_features = 200

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
filename_classifier = 'emo_models/svm_nostopwords_glove.sav'
pickle.dump(svm_linear_clf, open(filename_classifier, 'wb'))

# Saving the frequent keywords and label encoder
filename_freq_keywords = 'emo_models/nostopwords_frequent_keywords.sav'
pickle.dump(frequent_keywords, open(filename_freq_keywords, 'wb'))

filename_encoder = 'emo_models/glove_label_encoder.sav'
pickle.dump(label_encoder, open(filename_encoder, 'wb'))
