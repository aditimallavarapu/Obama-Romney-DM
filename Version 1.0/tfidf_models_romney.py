# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:42:46 2017

@author: Aditi and Suganya
"""

from CalculateMetrics import CalculateMetrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from Trainmodel import Naive_Bayes
import nltk
 
def draw_bar_plot(accuracy_list, fscore_list):
    objects = ('NB-Uni','NB-Bi','NB-Tri','MNB', 'BNB', 'KNN', 'DT', 'AB', 'RF', 'SVM', 'RBF', 'POLY')
    y_pos = np.arange(len(objects))
    bar_width = 0.35
     
    plt.bar(y_pos, accuracy_list, bar_width,
            alpha=0.5,
            color='b',
            label='Accuracy')
    plt.bar(y_pos + bar_width, fscore_list, bar_width, 
            alpha=0.5, 
            color='r', 
            label='FScore')
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
            overall_accuracy)
#            ,average_positive_precision,average_positive_recall,
#            average_positive_F1Score,average_negative_precision,average_positive_recall,
#            average_negative_F1Score)

calculate_metrics = CalculateMetrics()
tweets, tweetlist, labels = calculate_metrics.read_file("Romney_data_cleaned.txt")

number_of_folds = 10
subset_size = len(tweetlist)/number_of_folds
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


for gram in range(1,4):
    nb = Naive_Bayes("Romney_data_cleaned.txt",gram)
    training_set = nltk.classify.apply_features(nb.extract_features, nb.tweets)
    
    for i in range(number_of_folds):
        test_data_labels=[]
        test_set = training_set[i*subset_size:][:subset_size]
        for (text,label) in test_set:
            test_data_labels.append(label)
        train_set = training_set[:i*subset_size]+training_set[(i+1)*subset_size:]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        results = classifier.classify_many([fs for (fs, l) in test_set])
        if gram==1:
            nb_report_uni.append(classification_report(test_data_labels, results))
            nb_accuracy_list_uni.append(accuracy_score(test_data_labels, results))        
        elif gram==2:
            nb_report_bi.append(classification_report(test_data_labels, results))
            nb_accuracy_list_bi.append(accuracy_score(test_data_labels, results))
        elif gram==3:
             nb_report_tri.append(classification_report(test_data_labels, results))
             nb_accuracy_list_tri.append(accuracy_score(test_data_labels, results))

for i in range(number_of_folds):
     test_data = tweetlist[i*subset_size:][:subset_size]
     train_data = tweetlist[:i*subset_size] + tweetlist[(i+1)*subset_size:]
     test_data_labels = labels[i*subset_size:][:subset_size]
     train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]
 
      
 
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=6,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=False)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     mnb_classifier = MultinomialNB(alpha=0).fit(tfidf_matrix, train_data_labels)
     predictions = mnb_classifier.predict(test_tfidf_matrix)
     mnb_report.append(classification_report(test_data_labels, predictions))
     mnb_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
 
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=7,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     bnb_classifier = BernoulliNB(alpha=1.06).fit(tfidf_matrix, train_data_labels)
     predictions = bnb_classifier.predict(test_tfidf_matrix)
     bnb_report.append(classification_report(test_data_labels, predictions))
     bnb_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1), min_df=31,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     knn_classifier = KNeighborsClassifier(n_neighbors=9).fit(tfidf_matrix, train_data_labels)
     predictions = knn_classifier.predict(test_tfidf_matrix)
     knn_report.append(classification_report(test_data_labels, predictions))
     knn_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=15,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     dt_classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, train_data_labels)
     predictions = dt_classifier.predict(test_tfidf_matrix)
     dt_report.append(classification_report(test_data_labels, predictions))
     dt_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=15,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     ab_classifier = AdaBoostClassifier(n_estimators=57).fit(tfidf_matrix, train_data_labels)
     predictions = ab_classifier.predict(test_tfidf_matrix)
     ab_report.append(classification_report(test_data_labels, predictions))
     ab_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
          
     rf_classifier = RandomForestClassifier(n_estimators=35).fit(tfidf_matrix, train_data_labels)
     predictions = rf_classifier.predict(test_tfidf_matrix)
     rf_report.append(classification_report(test_data_labels, predictions))
     rf_accuracy_list.append(accuracy_score(test_data_labels, predictions))

     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     svm_classifier = svm.LinearSVC(C=0.35).fit(tfidf_matrix, train_data_labels)
     predictions = svm_classifier.predict(test_tfidf_matrix)
     linear_svm_report.append(classification_report(test_data_labels, predictions))
     linear_svm_accuracy_list.append(accuracy_score(test_data_labels, predictions))
#    
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     rbf_classifier = svm.SVC(kernel = 'rbf',gamma=0.75,decision_function_shape='ovo').fit(tfidf_matrix, train_data_labels)
     predictions = rbf_classifier.predict(test_tfidf_matrix)
     rbf_svm_report.append(classification_report(test_data_labels, predictions))
     rbf_svm_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     
     tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
     tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
     test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
     
     poly_classifier = svm.SVC(kernel = 'poly',degree=6,decision_function_shape='ovr').fit(tfidf_matrix, train_data_labels)
     predictions = poly_classifier.predict(test_tfidf_matrix)
     poly_svm_report.append(classification_report(test_data_labels, predictions))
     poly_svm_accuracy_list.append(accuracy_score(test_data_labels, predictions))
     print("Fold " + str(i) + " complete")
 

print("\nNaive Bayes Unigram")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(nb_report_uni, nb_accuracy_list_uni)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nNaive Bayes Bigram")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(nb_report_bi, nb_accuracy_list_bi)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nNaive Bayes Trigram")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(nb_report_tri, nb_accuracy_list_tri)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nMultinomial Naive Bayes Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(mnb_report, mnb_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nBernoulli Naive Bayes Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(bnb_report, bnb_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nK Nearest Neighbors Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(knn_report, knn_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nDecision Trees Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(dt_report, dt_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nAdaBoost Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(ab_report, ab_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nRandom Forest Classifier")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(rf_report, rf_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nSupport Vector Machines")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(linear_svm_report, linear_svm_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nRBF Kernel SVM")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(rbf_svm_report, rbf_svm_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

print("\nPolynomial Kernel SVM")
average_precision, average_recall, average_fscore, overall_accuracy = print_metrics(poly_svm_report, poly_svm_accuracy_list)
precision_list.append(average_precision)
recall_list.append(average_recall)
fscore_list.append(average_fscore)
accuracy_list.append(overall_accuracy)

draw_bar_plot(accuracy_list, fscore_list)
