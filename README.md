# hlt-report
This repository contains my final report for the Text Mining course Introduction to Human Language Technology.

Four systems were created by using combinations of two different topic detectors and two different emotion detectors and were evaluated on how well they could detect the topics and emotions present in a test set consisting of sentences.

The repository includes the code to train the classifiers, (`train_clas_glove_nostopwords.py`, `train_clas_glove_stopwords.py`, `train_clas_gn_nostopwords.py`, `train_clas_gn_stopwords.py`), the code for evaluation of the classifiers (`evaluation.py`) and files that contain all functions used (`utils.py` and `main.py`). In addition, the trained models and their additional files are stored in the `emo_models` folder and the results from the evaluation is stored in the `responses` folder.
