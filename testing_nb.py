# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:58:40 2017

@author: Aditi
"""
import nltk
import pickle
from nb_Romney import TwitterSentimentAnalysis
from preprocess import Preprocess

pre= Preprocess()
pre.xls_to_txt('testing-Obama-Romney-tweets.xlsx','test_obama.txt','test_romney.txt')
print("Text file saved")           
pre.clean_sets('test_romney_cleaned.txt','test_romney.txt','test_obama_cleaned.txt','test_obama.txt')

romney_model = TwitterSentimentAnalysis()
tweets = romney_model.read_file("test_romney.txt",2)
word_features = romney_model.get_word_features(romney_model.get_words_in_tweets(tweets))
test_set = nltk.classify.apply_features(romney_model.extract_features, tweets)
print("Starting...")
#classifier = nltk.NaiveBayesClassifier.train(training_set)

f = open('unigram_Romeny_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("compae.txt","wb")
for record,actual in test_set:
    linecount=linecount+1
    predict= classifier.classify(record)
    print "predic:" ,predict
    print "actual:" ,actual
    f.write(str(record))
    f.write("\t")
    f.write(actual)
    f.write("\t")
    f.write(predict)
    if(int(actual) is int(predict)):
        count=count+1
f.close()         
accu = float(count)/float(linecount)
        
#f = open("compae.txt","r")
#
#for line in f.readlines():
#
#    record = line.split("\t")
#           
#    if(str(record[1])==str(record[2])):
#        print record[1], record[2]  
#        print "match"
#        count = count+1;
#        print "count",count
#f.close()

print accu
print float(nltk.classify.accuracy(classifier, test_set))

##precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = romney_model.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = romney_model.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = romney_model.calculate_metrics(classifier,test_set, '-1')
##print(sum(tp_class1))
##print(sum(fp_class1))
##print(sum(tn_class1))
##print(sum(fn_class1))
#print('Class 1')
#print('Precision ', precision_class1)
#print('Recall ' , recall_class1)
#print('F1Score ' , f1score_class1)
#print('Class 0')
#print('Precision ', precision_class0)
#print('Recall ' , recall_class0)
#print('F1Score ' , f1score_class0)
#print('Class -1')
#print('Precision ', precision_class_negative)
#print('Recall ' , recall_class_negative)
#print('F1Score ' , f1score_class_negative)
#"""print('Class 2')
#print('Precision ', precision_class2)
#print('Recall ' , recall_class2)
#print('F1Score ' , f1score_class2)"""
#print('Overall Test Accuracy ', accuracy)