from utils import read_qa, load_classifier, load_semantic_model, create_response

# Adjusted function to get a response from a design
def get_response(last_message, emb_type, clas_type, print_info = False):
    """
    Loads all data needed to create a response and creates it. User can choose
    which embedding type and classifier will be used
    """
    # Load data for responses
    qa_data = read_qa(qa_path = 'data/qa_data/assignment_data.json')

    # Load embedding model
    embedding_model = load_semantic_model(model_type = emb_type)

    # Load classifier
    classifier = load_classifier(classifier_type = clas_type, model_type = emb_type)

    # Generate response
    response, emotion, topic, word_intersection = create_response(last_message,
                                                           qa_data,
                                                           classifier,
                                                           embedding_model, emb_type,
                                                           clas_type)
    # Print info if wanted
    if print_info:
        print()
        print("Received: {message}".format(message=last_message))
        print("Responded: {response}".format(response=response))
        print("Topic detected: {topic}".format(topic=topic))
        print("Emotion detected: {emotion}".format(emotion=emotion))
        print("Keywords detected [(keyword): (message_token)]: \n\t{intersection}".format(intersection=word_intersection))

    return response, emotion, topic, word_intersection

last_message = "I hate dogs!"

get_response(last_message, emb_type = 'gn', clas_type = 'stopwords', print_info = True)
