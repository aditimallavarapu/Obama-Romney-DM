# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:32:10 2017

@author: Aditi
"""

from preprocess import Preprocess
from Trainmodel import TwitterSentimentAnalysis
import nltk
import pickle

"""preprocess all training files"""
process = Preprocess() 
#script_dir = os.path.dirname("") #<-- absolute dir the script is in
process.xls_to_txt('training-Obama-Romney-tweets.xlsx','Obama_data.txt','Romney_data.txt')
print("Text file saved")           
process.clean_text_files('Romney_data_cleaned.txt','Romney_data.txt','Obama_data_cleaned.txt','Obama_data.txt')

"""Unigram Obama Model"""
analysis = TwitterSentimentAnalysis("Obama_data_cleaned.txt",1)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Unigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('unigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bigram Obama Model"""
analysis = TwitterSentimentAnalysis("Obama_data_cleaned.txt",2)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Bigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Bigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Trigram Obama Model"""
analysis = TwitterSentimentAnalysis("Obama_data_cleaned.txt",3)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...trigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('trigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Quadgram Obama Model"""
analysis = TwitterSentimentAnalysis("Obama_data_cleaned.txt",4)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...quadgram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Quadgram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Unigram Romney Model"""
analysis = TwitterSentimentAnalysis("Romney_data_cleaned.txt",1)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Unigram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('unigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bigram Romney Model"""
analysis = TwitterSentimentAnalysis("Romney_data_cleaned.txt",2)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Bigram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Bigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Trigram Romney Model"""
analysis = TwitterSentimentAnalysis("Romney_data_cleaned.txt",3)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...trigram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('trigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Quadgram Romney Model"""
analysis = TwitterSentimentAnalysis("Obama_data_cleaned.txt",4)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...quadgram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Quadgram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
