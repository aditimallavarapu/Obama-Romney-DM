# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:24:42 2017

@author: Aditi
"""
import os 
import nltk
from nb_Romney import TwitterSentimentAnalysis
import pickle

analysis = TwitterSentimentAnalysis()
script_dir = os.path.dirname("") #<-- absolute dir the script is in
"""Read actual file have file name here"""
analysis.xls_to_txt('training-Obama-Romney-tweets.xlsx')
print("Text file saved")
tweets = analysis.read_file("Romney_data.txt")

word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(analysis.extract_features, tweets)
#third = int(float(len(training_set)) / 3.0)
#print third
#train_set = training_set[0:(2*third)]
test_set = training_set[(2*third+1):]
training_set=[]
#print training_set
print("Starting...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

f = open('bigram_Romeny_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

#precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(test_set, '0')
#precision_class2, recall_class2, f1score_class2, accuracy = analysis.calculate_metrics(test_set, '2')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(test_set, '-1')
#print(sum(tp_class1))
#print(sum(fp_class1))
#print(sum(tn_class1))
#print(sum(fn_class1))
print('Class 1')
print('Precision ', precision_class1)
print('Recall ' , recall_class1)
print('F1Score ' , f1score_class1)
print('Class 0')
print('Precision ', precision_class0)
print('Recall ' , recall_class0)
print('F1Score ' , f1score_class0)
print('Class -1')
print('Precision ', precision_class_negative)
print('Recall ' , recall_class_negative)
print('F1Score ' , f1score_class_negative)
"""print('Class 2')
print('Precision ', precision_class2)
print('Recall ' , recall_class2)
print('F1Score ' , f1score_class2)"""
print('Overall Test Accuracy ', accuracy)

#print nltk.classify.accuracy(classifier, train_set)
#print nltk.classify.accuracy(classifier, test_set)
#print classifier.show_most_informative_features(32)
print("Model built...")
#tweet = '4 ppl being killed in a terrorist attack in Libya, Obama  is busy fundraising.'
#print classifier.classify(extract_features(analysis.generate_input_tokens(1, analysis.cleanup(tweet)))) 
