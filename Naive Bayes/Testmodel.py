# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:58:40 2017

@author: Aditi
"""
import nltk
import pickle
from Trainmodel import Naive_Bayes
from preprocess import Preprocess

pre= Preprocess()
pre.xls_to_txt('testing-Obama-Romney-tweets.xlsx','test_obama.txt','test_romney.txt')
print("Text file saved")           
pre.clean_text_files('test_romney_cleaned.txt','test_romney.txt','test_obama_cleaned.txt','test_obama.txt')

"""Unigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",1)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Unigram Obama")
f = open('unigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Unigram_obama.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
        

print "accuracy: ",accu
#print float(nltk.classify.accuracy(classifier, test_set))

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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

"""Bigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",2)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Bigram Obama")
f = open('Bigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Bigram_obama.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
print "accuracy:",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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


"""Trigram Obama Model"""
analysis = TwitterSentimentAnalysis("test_obama_cleaned.txt",3)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Trigram Obama")
f = open('trigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Trigram_obama.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
print "accuracy:",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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


"""Quadgram Obama Model"""
analysis = TwitterSentimentAnalysis("test_obama_cleaned.txt",4)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... quadigram Obama")
f = open('Quadgram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Quadgram_obama.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
        

print "accuracy",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
print('Overall Test Accuracy ', accuracy)


"""Unigram Romney Model"""
analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",1)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Unigram romney")
f = open('unigram_Romney_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Unigram_romney.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
        

print "accuracy: ",accu
#print float(nltk.classify.accuracy(classifier, test_set))

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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

"""Bigram Romney Model"""
analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",2)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Bigram Romney")
f = open('Bigram_Romney_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Bigram_romney.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
print "accuracy:",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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


"""Trigram Romney Model"""
analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",3)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Trigram Romney")
f = open('trigram_Romney_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Trigram_romney.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
print "accuracy:",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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


"""Quadgram Romney Model"""
analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",4)
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... quadigram romney")
f = open('Quadgram_Romney_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
count=0
linecount =0
f =open("Quadgram_romney.txt","wb")
for record,actual in test_set:
    if(actual.strip() != "ir"):
        if(int(actual) != 2):
            linecount=linecount+1
            predict= classifier.classify(record)
            f.write(str(record))
            f.write("\t")
            f.write(actual)
            f.write("\t")
            f.write(predict)
            if(int(actual) is int(predict)):
                count=count+1
f.close()         
accu = float(count)/float(linecount)
        

print "accuracy",accu

##precision, recall, F1score
precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
print('Overall Test Accuracy ', accuracy)


