# evaluation

from main import get_response
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
import json

"""
RESPONSES
"""

# The test sentences and labels are saved in a dictionary for comparison
with open("test_set.txt", 'r') as tsvfile:
    messages = dict()
    for n, line in enumerate(tsvfile):
        values = line.strip()
        values = values.split('\t')
        current_message = dict()
        current_message['message_txt'] = values[0]
        current_message['topic'] = values[1]
        current_message['emotion'] = values[2]
        messages[n] = current_message

# The test dictionary is saved as a json file for later use
with open("responses/messages.json", 'w') as outfile:
    json.dump(messages, outfile)

# For each design, responses for all test sentences are saved along with preditions for topic and emotions

print("GLOVE AND NOSTOPWORDS")
responses_glove_nostopwords = dict()

# For each test sentence, a response is created and saved with the predicted topic and emotion
for n in messages:
    current_response = dict()
    message = messages[n]['message_txt']
    print(n+1)
    response, emotion, topic, word_intersection = get_response(message, emb_type = 'glove', clas_type = 'nostopwords', print_info = True)
    current_response['response_txt'] = response
    current_response['topic'] = topic
    current_response['emotion'] = emotion

    # Repsonse and predictions are saved in a dictionary
    responses_glove_nostopwords[n] = current_response

# The dictionary with repsonses and predictions is saved a json file for later use
with open("responses/responses_glove_nostopwords.json", "w") as outfile:
     json.dump(responses_glove_nostopwords, outfile)

# The responses are saved as txt file for later use
with open("responses/responses_glove_nostopwords.txt", 'w') as outfile:
    for n in responses_glove_nostopwords:
        outfile.write(f"{responses_glove_nostopwords[n]['response_txt']}\n")

# GLOVE AND STOPWORDS
print("GLOVE AND STOPWORDS")
responses_glove_stopwords = dict()
for n in messages:
    current_response = dict()
    message = messages[n]['message_txt']
    print(n+1)
    response, emotion, topic, word_intersection = get_response(message, emb_type = 'glove', clas_type = 'stopwords', print_info = True)
    current_response['response_txt'] = response
    current_response['topic'] = topic
    current_response['emotion'] = emotion
    responses_glove_stopwords[n] = current_response

with open("responses/responses_glove_stopwords.json", "w") as outfile:
     json.dump(responses_glove_stopwords, outfile)

with open("responses/responses_glove_stopwords.txt", 'w') as outfile:
    for n in responses_glove_stopwords:
        outfile.write(f"{responses_glove_stopwords[n]['response_txt']}\n")


# GN AND NOSTOPWORDS
print("GN AND NOSTOPWORDS")
responses_gn_nostopwords = dict()
for n in messages:
    current_response = dict()
    message = messages[n]['message_txt']
    print(n+1)
    response, emotion, topic, word_intersection = get_response(message, emb_type = 'gn', clas_type = 'nostopwords', print_info = True)
    current_response['response_txt'] = response
    current_response['topic'] = topic
    current_response['emotion'] = emotion
    responses_gn_nostopwords[n] = current_response

with open("responses/responses_gn_nostopwords.json", "w") as outfile:
     json.dump(responses_gn_nostopwords, outfile)

with open("responses/responses_gn_nostopwords.txt", 'w') as outfile:
    for n in responses_gn_nostopwords:
        outfile.write(f"{responses_gn_nostopwords[n]['response_txt']}\n")


# GN AND STOPWORDS
print("GN AND STOPWORDS")
responses_gn_stopwords = dict()
for n in messages:
    current_response = dict()
    message = messages[n]['message_txt']
    print(n+1)
    response, emotion, topic, word_intersection = get_response(message, emb_type = 'gn', clas_type = 'stopwords', print_info = True)
    current_response['response_txt'] = response
    current_response['topic'] = topic
    current_response['emotion'] = emotion
    responses_gn_stopwords[n] = current_response

with open("responses/responses_gn_stopwords.json", "w") as outfile:
     json.dump(responses_gn_stopwords, outfile)

with open("responses/responses_gn_stopwords.txt", 'w') as outfile:
    for n in responses_gn_stopwords:
        outfile.write(f"{responses_gn_stopwords[n]['response_txt']}\n")


"""
RECALL & PRECISION
"""
# Label encoding
emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
topics = ['animals', 'people', 'sports', 'food', 'places', 'unknown']

# The test sentences are loaded in
with open("responses/messages.json", 'r') as infile:
    messages = json.load(infile)

gold_emotions = []
gold_topics = []

# The gold topic and emotion labels are prepared for comparison
for n in messages:
    gold_topic = messages[n]['topic']
    gold_index = topics.index(gold_topic)
    gold_topics.append(gold_index)

    gold_emotion = messages[n]['emotion']
    gold_index = emotions.index(gold_emotion)
    gold_emotions.append(gold_index)

# For each design, the topic and emotion labels are prepared for comparison
system_emotions = []
system_topics = []

with open("responses/responses_glove_nostopwords.json", 'r') as infile:
    responses_glove_nostopwords = json.load(infile)

for n in responses_glove_nostopwords:
    system_topic = responses_glove_nostopwords[n]['topic']
    system_index = topics.index(system_topic)
    system_topics.append(system_index)

    system_emotion = responses_glove_nostopwords[n]['emotion']
    system_index = emotions.index(system_emotion)
    system_emotions.append(system_index)

# Classification reports are generated for both topic and emotion
topic_report = classification_report(gold_topics, system_topics, digits = len(topics))

emotion_report = classification_report(gold_emotions, system_emotions, digits = len(emotions))

print("Glove and NOSTOPWORDS:")
print(topics)
print(topic_report)
print(emotions)
print(emotion_report)

# GLOVE & STOPWORDS
system_emotions = []
system_topics = []

with open("responses/responses_glove_stopwords.json", 'r') as infile:
    responses_glove_stopwords = json.load(infile)

for n in responses_glove_stopwords:

    system_topic = responses_glove_stopwords[n]['topic']
    system_index = topics.index(system_topic)
    system_topics.append(system_index)

    system_emotion = responses_glove_stopwords[n]['emotion']
    system_index = emotions.index(system_emotion)
    system_emotions.append(system_index)

topic_report = classification_report(gold_topics, system_topics, digits = len(topics))

emotion_report = classification_report(gold_emotions, system_emotions, digits = len(emotions))

print("Glove and STOPWORDS:")
print(topics)
print(topic_report)
print(emotions)
print(emotion_report)

# GN & NOSTOPWORDS
system_emotions = []
system_topics = []

with open("responses/responses_gn_nostopwords.json", 'r') as infile:
    responses_gn_nostopwords = json.load(infile)

for n in responses_gn_nostopwords:

    system_topic = responses_gn_nostopwords[n]['topic']
    system_index = topics.index(system_topic)
    system_topics.append(system_index)

    system_emotion = responses_gn_nostopwords[n]['emotion']
    system_index = emotions.index(system_emotion)
    system_emotions.append(system_index)

topic_report = classification_report(gold_topics, system_topics, digits = len(topics))

emotion_report = classification_report(gold_emotions, system_emotions, digits = len(emotions))

print("Gn and NOSTOPWORDS:")
print(topics)
print(topic_report)
print(emotions)
print(emotion_report)

# GN & STOPWORDS
system_emotions = []
system_topics = []

with open("responses/responses_gn_stopwords.json", 'r') as infile:
    responses_gn_stopwords = json.load(infile)

for n in responses_gn_stopwords:

    system_topic = responses_gn_stopwords[n]['topic']
    system_index = topics.index(system_topic)
    system_topics.append(system_index)

    system_emotion = responses_gn_stopwords[n]['emotion']
    system_index = emotions.index(system_emotion)
    system_emotions.append(system_index)

topic_report = classification_report(gold_topics, system_topics, digits = len(topics))

emotion_report = classification_report(gold_emotions, system_emotions, digits = len(emotions))

print(system_topics)
print(system_emotions)

print("Gn and STOPWORDS:")
print(topics)
print(topic_report)
print(emotions)
print(emotion_report)
