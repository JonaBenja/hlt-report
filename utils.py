import json
import requests
import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from nltk.corpus import stopwords
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
import random
from collections import defaultdict

def read_qa(qa_path):
    with open(qa_path) as f:
        qa_data = json.load(f)

    return qa_data

"""
VECTORIZE DATA
"""

def tokenize_data(text):
    ### the first loop gets the utterances
    text_tokens = []
    for utterance in text:
        text_tokens.append(nltk.tokenize.word_tokenize(utterance))

    return text_tokens

def featureVecMethod(words, # Tokenized list of tokens from an utterance
                     frequent_keywords, # List of words above the frequency threshold
                     stop_words, # Stopwords that should be skipped
                     model, # The actual word embeddings model
                     modelword_index, # An index on the vocabulary of the model to speed up lookup
                     num_features # the number of dimensions of the embedding model
                    ):
    featureVec = np.zeros(num_features,dtype="float32")

    nwords = 0

    unknown_words = []
    known_words = []

    for word in  words:
        #### we only use words that are frequent and not stopwords
        if word in frequent_keywords and not word in stop_words:
            if word in modelword_index:
                featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))

                known_words.append(word)
                nwords = nwords + 1
            else:
                word = word.lower()
                if word in modelword_index:
                    featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))

                    known_words.append(word)
                    nwords = nwords + 1
                else:

                    unknown_words.append(word)

    featureVec = np.divide(featureVec, nwords)
    return featureVec

# Function for calculating the average feature vector
def getAvgFeatureVecs(texts,
                      keywords,
                      stopwords,
                      model,
                      modelword_index,
                      num_features
                     ):
    counter = 0
    #### we initialise a numpy vector with zeros of the type float32
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")

    #### We iterate over all the texts
    for text in texts:
        textFeatureVecs[counter] = featureVecMethod(text, keywords, stopwords, model, modelword_index,num_features)
        counter = counter+1

    textFeatureVecs = np.nan_to_num(textFeatureVecs)

    return textFeatureVecs


"""
LOADING EMBEDDING MODEL & CLASSIFIER
"""
def load_semantic_model(model_type):
    """ Function to load word embedding models needed """
    if model_type == 'glove':
        glove_file = datapath('/Users/jonabenja/Desktop/glove.twitter.27B/glove.twitter.27B.200d.txt')
        tmp_file = get_tmpfile('test_word2vec.txt')

        wordembeddings = glove2word2vec(glove_file, tmp_file)
        embedding_model = KeyedVectors.load_word2vec_format(tmp_file)

    elif model_type == 'leipzig':
        embedding_file = 'emb_models/leipzig_200.bin'
        embedding_model = KeyedVectors.load_word2vec_format(embedding_file, binary = True)

    elif model_type == 'gn':
        embedding_file = 'emb_models/GoogleNews-vectors-negative300.bin'
        embedding_model = KeyedVectors.load_word2vec_format(embedding_file, binary = True)

    return embedding_model

def load_classifier(classifier_type, model_type):
    """ Function to load pre-trained machine learning models needed """

    filename_classifier = f"emo_models/svm_{classifier_type}_{model_type}.sav"

    loaded_classifier = pickle.load(open(filename_classifier, 'rb'))

    return loaded_classifier

"""
TOPIC DETECTION
"""
def get_similar_words(embedding_model, message, num_similar_words=10, verbose=False):
    """ Function to enrich the message with similar words for better keyword detection """
    tokens = nltk.tokenize.word_tokenize(message)

    similar_words = defaultdict(set)
    for token in set(tokens):
        similar_words[token].add(token)

        try:
            word_neighborhood = embedding_model.most_similar(positive=[token], topn=num_similar_words)

            # Add neighbor words to enrich the message
            for item in word_neighborhood:
                word = item[0].lower()
                similar_words[word].add(token) # EDIT THIS TO TOKEN

        except KeyError as e:
            print("token '%s' not in embedding vocabulary" % token)

    if verbose:
        PrettyPrinter(indent=2).pprint(similar_words)

    return similar_words

def semantic_similarity(message, keywords):
    """ Function to determine if the message matches certain keywords according to some semantic similarity or relatedness"""
    # Get enriched tokens
    message_words = message.keys()

    # Calculate intersection between the two sets of words
    word_intersection = list(set(keywords) & set(message_words))

    # Create a dictionary so we know what keywords matched to what original token
    matched_words = {w: message[w] for w in word_intersection}

    return matched_words

"""
EMOTION CLASSIFICATION
"""
def classify_emotion(message, classifier, word_embedding_model, model_type, classifier_type):
    """ Function to process a message and predict the emotion it reflects """
    #message = [message]

    filename_encoder = f'emo_models/{model_type}_label_encoder.sav'

    loaded_label_encoder = pickle.load(open(filename_encoder, 'rb'))

    if classifier_type == 'nostopwords':
        stop_words = set(stopwords.words('english'))
    elif classifier_type == 'stopwords':
        stop_words = []

    frequent_keywords = pickle.load(open(f'emo_models/{classifier_type}_frequent_keywords.sav', 'rb'))

    index2word_set = set(word_embedding_model.index2word)

    num_features = word_embedding_model.vector_size

    message_tokens = word_tokenize(message)

    message_embedding_vectors = getAvgFeatureVecs(message_tokens, frequent_keywords, stop_words, word_embedding_model, index2word_set, num_features)

    predictions = classifier.predict(message_embedding_vectors)

    for predicted_label in predictions:
        predicted_emotion = loaded_label_encoder.classes_[predicted_label]

    return predicted_emotion

"""
RESPONSE GENERATION
"""

def create_response(message, qa, classifier, embedding_model, model_type, classifier_type):
    # Determine default response
    reply = "I'm afraid I cannot answer that."
    emotion = "unknown"
    topic = "unknown"

    # Classify emotion in message
    emotion = classify_emotion(message, classifier, embedding_model, model_type, classifier_type)

    # Enrich message
    similar_words = get_similar_words(embedding_model, message)

    # Loop through the predefined intents, and generate a response if there is a match (emotion + keywords)
    word_intersection = {}
    for i in qa['intents']:

        # Only consider intents related to the emotion detected
        if emotion == i['emotion']:

            # Try to match the message to the set of predefined keywords
            word_intersection = semantic_similarity(similar_words, keywords=i['keywords'])

            # If there is a match, generate a response response
            if word_intersection:
                topic = i['topic']
                break

    if emotion != 'unknown' and topic != 'unknown':
        appropriate_responses = list(filter(lambda i: emotion == i['emotion'] and topic == i['topic'], qa['intents']))
        reply = random.choice(appropriate_responses[0]['responses'])

    return reply, emotion, topic, word_intersection
