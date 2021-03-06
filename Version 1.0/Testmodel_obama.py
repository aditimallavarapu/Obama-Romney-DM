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
from sklearn.externals import joblib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
def draw_bar_plot(accuracy_list, fscore_list):
    objects = ('NB-Uni','NB-Bi','NB-tri','SVM','MNB', 'BNB', 'DT', 'AB', 'RF','RBF', 'KNN')
    y_pos = np.arange(len(objects))
    bar_width = 0.20
     
    plt.bar(y_pos, accuracy_list, bar_width,
            alpha=0.5,
            color='b',
            label='Accuracy')
    plt.bar(y_pos + bar_width, fscore_list, bar_width, 
            alpha=0.5, 
            color='r', 
            label='FScore')
    plt.bar(y_pos + (2*bar_width), positive_fscore_list, bar_width, 
            alpha=0.5, 
            color='g', 
            label='Positive Fscore')
    plt.bar(y_pos + (3*bar_width), negative_fscore_list, bar_width, 
            alpha=0.5, 
            color='y', 
            label= 'Negative Fscore')    
           
    plt.xticks(y_pos + bar_width, objects)
    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title('Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
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
            overall_accuracy,average_positive_precision,average_positive_recall,
            average_positive_F1Score,average_negative_precision,average_positive_recall,
            average_negative_F1Score)
            
            
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
positive_precision_list =[]
negative_precision_list=[]
positive_recall_list=[]
negative_recall_list=[]
positive_fscore_list=[]
negative_fscore_list=[]

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
        f.write(",")
        f.write(actual)
        f.write(",")
        f.write(predict)
        predict_list.append(predict)
    f.close()    
    return predict_list

def write_to_file_tfidf(f,classifier,tweets,labels):
    predictions= classifier.predict(tweets)
    predict_list=[]
    for (record,actual,predict) in zip(tweets, labels, predictions):
#        predict= classifier.predict(record)
        f.write("\n")
        f.write(str(record))
        f.write(",")
        f.write(actual)
        f.write(",")
        f.write(predict)
        predict_list.append(predict)
    f.close()    
    return predict_list        
        
    
"""Unigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",1)  

tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Unigram Obama")
f = open('unigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f =open("NB_Unigram_obama.csv","wb")
predict_list = write_to_file_nb(f,classifier,test_set,tweet_list)
nb_report_uni.append(classification_report(labels, predict_list))
nb_accuracy_list_uni.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(nb_report_uni, nb_accuracy_list_uni)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Bigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",2)  
metrics = CalculateMetrics()
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
labels=[]
for (text,label) in test_set:
    labels.append(label)
print("Bigram Obama")
f = open('Bigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f =open("NB_Bigram_obama.csv","wb")
predict_list = write_to_file_nb(f,classifier,test_set,tweet_list)
nb_report_bi.append(classification_report(labels, predict_list))
nb_accuracy_list_bi.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(nb_report_bi, nb_accuracy_list_bi)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Trigram Obama Model"""
analysis = Naive_Bayes("test_obama_cleaned.txt",3)  
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
test_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
labels=[]
for (text,label) in test_set:
    labels.append(label)
print("Trigram Obama")
f = open('Trigram_Obama_nb_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f =open("NB_Trigram_obama.csv","wb")
predict_list = write_to_file_nb(f,classifier,test_set,tweets)
nb_report_tri.append(classification_report(labels, predict_list))
nb_accuracy_list_tri.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(nb_report_tri, nb_accuracy_list_tri)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Linear SVM Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Linear SVM Obama")
classifier =joblib.load('LinearSVM_Obama_classifier.pickle')
f =open("LinearSVM_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
linear_svm_report.append(classification_report(labels, predict_list))
linear_svm_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(linear_svm_report, linear_svm_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Multinomial NB Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Multinomial NB Obama")
classifier =joblib.load('Multinomial_Obama_classifier.pickle')
f =open("MultinomialNB_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
mnb_report.append(classification_report(labels, predict_list))
mnb_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(mnb_report, mnb_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Bernouli NB Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Bernouli NB Obama")
classifier =joblib.load('Bernouli_Obama_classifier.pickle')
f =open("BernouliNB_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
bnb_report.append(classification_report(labels, predict_list))
bnb_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(bnb_report, bnb_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Decision Tree Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Decision Tree Obama")
classifier =joblib.load('Decision_tree_Obama_classifier.pickle')
f =open("Decisiontree_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
dt_report.append(classification_report(labels, predict_list))
dt_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(dt_report, dt_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Adaboost Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Adaboost Obama")
classifier =joblib.load('Adaboost_Obama_classifier.pickle')
f =open("Adaboost_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
ab_report.append(classification_report(labels, predict_list))
ab_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(ab_report, ab_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""Random Forests Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("Random Forest Obama")
classifier =joblib.load('Random_Forests_Obama_classifier.pickle')
f =open("Random_Forests_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
rf_report.append(classification_report(labels, predict_list))
rf_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(rf_report, rf_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""RBF SVM Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("RBF SVM Obama")
classifier =joblib.load('RBF_SVM_Obama_classifier.pickle')
f =open("RBF_SVM_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
rbf_svm_report.append(classification_report(labels, predict_list))
rbf_svm_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(rbf_svm_report, rbf_svm_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

"""KNN Obama Model"""
tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
print("KNN Obama")
classifier =joblib.load('KNN_Obama_classifier.pickle')
f =open("KNN_obama.csv","wb")
predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
knn_report.append(classification_report(labels, predict_list))
knn_accuracy_list.append(accuracy_score(labels, predict_list))        
precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(knn_report, knn_accuracy_list)
fscore_list.append(fscore)
accuracy_list.append(overall_accuracy)
positive_precision_list.append(positive_precision)
negative_precision_list.append(negative_precision)
positive_recall_list.append(positive_recall)
negative_recall_list.append(negative_recall)
positive_fscore_list.append(positive_fscore)
negative_fscore_list.append(negative_fscore)

#"""Polynomial SVM Model"""
#tweets,tweet_list,labels = metrics.read_file("test_obama_cleaned.txt")
#print("Polynomial SVM Obama")
#classifier =joblib.load('Polynomial_SVM_Obama_classifier.pickle')
#f =open("Polynomial_SVM_obama.csv","wb")
#predict_list = write_to_file_tfidf(f,classifier,tweet_list,labels)
#poly_svm_report.append(classification_report(labels, predict_list))
#poly_svm_accuracy_list.append(accuracy_score(labels, predict_list))        
#precision, recall, fscore, overall_accuracy,positive_precision,positive_recall, positive_fscore,negative_precision,negative_recall,negative_fscore = print_metrics(poly_svm_report, poly_svm_accuracy_list)
#fscore_list.append(fscore)
#accuracy_list.append(overall_accuracy)
#positive_precision_list.append(positive_precision)
#negative_precision_list.append(negative_precision)
#positive_recall_list.append(positive_recall)
#negative_recall_list.append(negative_recall)
#positive_fscore_list.append(positive_fscore)
#negative_fscore_list.append(negative_fscore)
#
#
draw_bar_plot(accuracy_list, fscore_list)
