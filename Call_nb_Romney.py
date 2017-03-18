# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:33:52 2017

@author: Aditi
"""
from nb_Romney import TwitterSentimentAnalysis
import nltk
import pickle

analysis = TwitterSentimentAnalysis()
tweets = analysis.read_file("Romney_data_cleaned.txt",2)
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(analysis.extract_features, tweets)
print("Starting...")
classifier = nltk.NaiveBayesClassifier.train(training_set)

f = open('unigram_Romeny_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

#tweets = analysis.read_file("test_romney.txt",1)
#word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
#test_set = nltk.classify.apply_features(analysis.extract_features, tweets)
#print("Starting...")
#classifier = nltk.NaiveBayesClassifier.train(training_set)

#f = open('unigram_Romeny_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()


        

