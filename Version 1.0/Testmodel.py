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
from sklearn.feature_extraction.text import TfidfVectorizer

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

metrics = CalculateMetrics()
pre= Preprocess()
pre.xls_to_txt('testing-Obama-Romney-tweets.xlsx','test_obama.txt','test_romney.txt')
print("Text file saved")           
pre.clean_text_files('test_romney_cleaned.txt','test_romney.txt','test_obama_cleaned.txt','test_obama.txt')
pre.remove_2("test_obama_cleaned.txt")
pre.remove_2("test_romney_cleaned.txt")

def write_to_file_nb(f,classifier,test_set,tweets):
    predict_list=[]
    for ((record,actual),tweet) in zip(test_set,tweets):
        predict= classifier.classify(record)
        f.write("\n")
        f.write(str(tweet))
        f.write("\t")
        f.write(actual)
        f.write("\t")
        f.write(predict)
        predict_list.append(predict)
    f.close()    
    return predict_list

def write_to_file_tfidf(f,classifier,test_tfidf_matrix,tweets,labels):
    predictions= classifier.predict(test_tfidf_matrix)
    predict_list=[]
    for (record,actual,predict) in zip(tweets, labels, predictions):
        #predict= classifier.predict(record)
        f.write("\n")
        f.write(str(record))
        f.write("\t")
        f.write(actual)
        f.write("\t")
        f.write(predict)
        predict_list.append(predict)
    f.close()    
    return predict_list        
        
    
#"""Unigram Obama Model"""
#analysis = Naive_Bayes("test_obama_cleaned.txt",1)  

#tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#print("Starting... Unigram Obama")
#f = open('unigram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#f =open("NB_Unigram_obama.txt","wb")
#predict_list = write_to_file_nb(f,classifier,test_set,tweet_list)
#nb_report_uni.append(classification_report(labels, predict_list))
#nb_accuracy_list_uni.append(accuracy_score(labels, predict_list))        
#precision, recall, fscore, overall_accuracy = print_metrics(nb_report_uni, nb_accuracy_list_uni)
#
#"""Bigram Obama Model"""
#analysis = Naive_Bayes("test_obama_cleaned.txt",2)  
#metrics = CalculateMetrics()
#tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#labels=[]
#for (text,label) in test_set:
#    labels.append(label)
#print("Starting... Bigram Obama")
#f = open('Bigram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#f =open("NB_Bigram_obama.txt","wb")
#predict_list = write_to_file_nb(f,classifier,test_set,tweet_list)
#nb_report_bi.append(classification_report(labels, predict_list))
#nb_accuracy_list_bi.append(accuracy_score(labels, predict_list))        
#precision, recall, fscore, overall_accuracy = print_metrics(nb_report_bi, nb_accuracy_list_bi)
#
#"""Trigram Obama Model"""
#analysis = Naive_Bayes("test_obama_cleaned.txt",3)  
#tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
#test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
#labels=[]
#for (text,label) in test_set:
#    labels.append(label)
#print("Starting... Trigram Obama")
#f = open('Trigram_Obama_nb_classifier.pickle', 'rb')
#classifier = pickle.load(f)
#f.close()
#f =open("NB_Trigram_obama.txt","wb")
#predict_list = write_to_file_nb(f,classifier,test_set,tweets)
#nb_report_tri.append(classification_report(labels, predict_list))
#nb_accuracy_list_tri.append(accuracy_score(labels, predict_list))        
#precision, recall, fscore, overall_accuracy = print_metrics(nb_report_tri, nb_accuracy_list_tri)
#
"""Linear SVM Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Starting...  Linear SVM Obama")
f = open('LinearSVM_Obama_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f =open("LinearSVM_obama.txt","wb")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
test_tfidf_matrix = tfidf_vectorizer.transform(tweet_list)
predict_list = write_to_file_tfidf(f,classifier,test_tfidf_matrix,tweet_list,labels)
linear_svm_report.append(classification_report(labels, predict_list))
linear_svm_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy = print_metrics(linear_svm_report, linear_svm_accuracy_list)