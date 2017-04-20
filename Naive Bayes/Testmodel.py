# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:58:40 2017

@author: Aditi
"""
import nltk
import pickle
from Trainmodel import Naive_Bayes
from preprocess import Preprocess
from CalculateMetrics import CalculateMetrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def evaluation_metrics(clasification_report_list):
    precision_list = []
    recall_list = []
    fscore_list = []
    positive_recall_list =[]
    positive_F1Score_list =[]    
    positive_precision_list =[]
    negative_recall_list =[]
    negative_F1Score_list =[]    
    negative_precision_list =[]
    for clasification_report in clasification_report_list:
        lines = clasification_report.split('\n')
        positive = lines[4].split()
        positive_precision_list.append(float(positive[1]))
        positive_recall_list.append(float(positive[2]))
        positive_F1Score_list.append(float(positive[3]))
        negative = lines[2].split()
        negative_precision_list.append(float(negative[1]))
        negative_recall_list.append(float(negative[2]))
        negative_F1Score_list.append(float(negative[3]))
        average_metrics = lines[6].split()
        precision_list.append(float(average_metrics[3]))
        recall_list.append(float(average_metrics[4]))
        fscore_list.append(float(average_metrics[5]))
    return (float(sum(precision_list))/len(precision_list), 
            float(sum(recall_list))/len(recall_list),
            float(sum(fscore_list))/len(fscore_list), 
            float(sum(positive_precision_list))/len(positive_precision_list),
            float(sum(positive_recall_list))/len(positive_recall_list),
            float(sum(positive_F1Score_list))/len(positive_F1Score_list),
            float(sum(negative_precision_list))/len(negative_precision_list),
            float(sum(negative_recall_list))/len(negative_recall_list),
            float(sum(negative_F1Score_list))/len(negative_F1Score_list)    )
                    
   
        
def print_metrics(clasification_report_list, accuracy_score_list):
    average_precision, average_recall, average_fscore, average_positive_precision, average_positive_recall, average_positive_F1Score, average_negative_precision, average_negative_recall, average_negative_F1Score  = evaluation_metrics(clasification_report_list)
    overall_accuracy = float(sum(accuracy_score_list))/len(accuracy_score_list)
    print("Average Precision: ", average_precision)
    print("Average Recall: ", average_recall)
    print("Average Fscore: ", average_fscore)
    print("Overall accuracy: ",overall_accuracy)
    print("Positive Precision: ", average_positive_precision)
    print("Positive Recall: ", average_positive_recall)
    print("Positive FScore: ", average_positive_F1Score)
    print("Negative Precision: ", average_negative_precision)    
    print("Negative Recall: ", average_negative_recall)
    print("Negative FScore: ", average_negative_F1Score)
    return (average_precision, average_recall, average_fscore, 
            overall_accuracy)
#            ,average_positive_precision,average_positive_recall,
#            average_positive_F1Score,average_negative_precision,average_positive_recall,
#            average_negative_F1Score)
            
            
mnb_report = []
gnb_report = []
bnb_report = []
knn_report = []
dt_report = []
rf_report = []
ab_report = []
nb_report_uni =[]
nb_report_bi =[]
nb_report_tri =[]
linear_svm_report = []
rbf_svm_report = []
poly_svm_report = []
mnb_accuracy_list = []
gnb_accuracy_list = []
bnb_accuracy_list = []
knn_accuracy_list = []
dt_accuracy_list = []
rf_accuracy_list = []
ab_accuracy_list = []
nb_accuracy_list_uni=[]
nb_accuracy_list_bi=[]
nb_accuracy_list_tri=[]
linear_svm_accuracy_list = []
rbf_svm_accuracy_list = []
poly_svm_accuracy_list = []
precision_list = []
recall_list = []
fscore_list = []
accuracy_list = []


pre= Preprocess()
pre.xls_to_txt('testing-Obama-Romney-tweets.xlsx','test_obama.txt','test_romney.txt')
print("Text file saved")           
pre.clean_text_files('test_romney_cleaned.txt','test_romney.txt','test_obama_cleaned.txt','test_obama.txt')
pre.remove_2("test_obama_cleaned.txt")
pre.remove_2("test_romney_cleaned.txt")

def write_to_file(f,classifier,test_set,tweets):
    predict_list=[]
    for ((record,actual),tweets) in zip(test_set,test_set):
        predict= classifier.classify(record)
        f.write(str(tweets))
        f.write("\t")
        f.write(actual)
        f.write("\t")
        f.write(predict)
        predict_list.append(predict)
    f.close()    
    return predict_list
        
        
    
"""Unigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",1)  
metrics = CalculateMetrics()
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned")
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting... Unigram Obama")
f = open('unigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f =open("NB_Unigram_obama.txt","wb")
predict_list = write_to_file(f,classifier,test_set,tweets)
nb_report_uni.append(classification_report(labels, predict_list))
nb_accuracy_list_uni.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy = print_metrics(nb_report_uni, nb_accuracy_list_uni)


#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
        

#print "accuracy: ",accu
##print float(nltk.classify.accuracy(classifier, test_set))
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#"""Bigram Obama Model"""
#analysis = Naive_Bayes("test_obama_cleaned.txt",2)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Bigram Obama")
#f = open('Bigram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Bigram_obama.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#print "accuracy:",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#
#"""Trigram Obama Model"""
#analysis = TwitterSentimentAnalysis("test_obama_cleaned.txt",3)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Trigram Obama")
#f = open('trigram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Trigram_obama.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#print "accuracy:",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#
#"""Quadgram Obama Model"""
#analysis = TwitterSentimentAnalysis("test_obama_cleaned.txt",4)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... quadigram Obama")
#f = open('Quadgram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Quadgram_obama.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#        
#
#print "accuracy",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#print('Overall Test Accuracy ', accuracy)
#
#
#"""Unigram Romney Model"""
#analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",1)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Unigram romney")
#f = open('unigram_Romney_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Unigram_romney.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#        
#
#print "accuracy: ",accu
##print float(nltk.classify.accuracy(classifier, test_set))
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#"""Bigram Romney Model"""
#analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",2)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Bigram Romney")
#f = open('Bigram_Romney_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Bigram_romney.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#print "accuracy:",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#
#"""Trigram Romney Model"""
#analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",3)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Trigram Romney")
#f = open('trigram_Romney_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Trigram_romney.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#print "accuracy:",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#
#
#"""Quadgram Romney Model"""
#analysis = TwitterSentimentAnalysis("test_romney_cleaned.txt",4)
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... quadigram romney")
#f = open('Quadgram_Romney_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#count=0
#linecount =0
#f =open("Quadgram_romney.txt","wb")
#for record,actual in test_set:
#    if(actual.strip() != "ir"):
#        if(int(actual) != 2):
#            linecount=linecount+1
#            predict= classifier.classify(record)
#            f.write(str(record))
#            f.write("\t")
#            f.write(actual)
#            f.write("\t")
#            f.write(predict)
#            if(int(actual) is int(predict)):
#                count=count+1
#f.close()         
#accu = float(count)/float(linecount)
#        
#
#print "accuracy",accu
#
###precision, recall, F1score
#precision_class1, recall_class1, f1score_class1, accuracy = analysis.calculate_metrics(classifier, test_set, '1')
#precision_class0, recall_class0, f1score_class0, accuracy = analysis.calculate_metrics(classifier,test_set, '0')
#precision_class_negative, recall_class_negative, f1score_class_negative, accuracy = analysis.calculate_metrics(classifier,test_set, '-1')
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
#print('Overall Test Accuracy ', accuracy)
#
#
