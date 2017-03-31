# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:42:46 2017

@author: Suganya
"""

from CalculateMetrics import CalculateMetrics
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def evaluation_metrics(precisionlist0, precisionlist1, precisionlist_1, recalllist0, recalllist1, recalllist_1, f1scorelist0, f1scorelist1, f1scorelist_1, accuracylist):
    precision0, recall0, f1score0, class_accuracy0, overall_accuracy = calculate_metrics.metrics("0\n", predictions, test_data_labels)
    precision1, recall1, f1score1, class_accuracy1, overall_accuracy = calculate_metrics.metrics("1\n", predictions, test_data_labels)
    precision_1, recall_1, f1score_1, class_accuracy_1, overall_accuracy = calculate_metrics.metrics("-1\n", predictions, test_data_labels)
    precisionlist0.append(precision0)
    precisionlist1.append(precision1)
    precisionlist_1.append(precision_1)
    recalllist0.append(recall0)
    recalllist1.append(recall1)
    recalllist_1.append(recall_1)
    f1scorelist0.append(f1score0)
    f1scorelist1.append(f1score1)
    f1scorelist_1.append(f1score_1)
    accuracylist.append((class_accuracy0 + class_accuracy1 + class_accuracy_1) / 3)
    return precisionlist0, precisionlist1, precisionlist_1, recalllist0, recalllist1, recalllist_1, f1scorelist0, f1scorelist1, f1scorelist_1, accuracylist

def print_values(precisionlist0, precisionlist1, precisionlist_1, recalllist0, recalllist1, recalllist_1, f1scorelist0, f1scorelist1, f1scorelist_1, accuracylist):
    print("Precision - Class 0", sum(precisionlist0) / float(len(precisionlist0)))
    print("Precision - Class 1", sum(precisionlist1) / float(len(precisionlist1)))
    print("Precision - Class -1", sum(precisionlist_1) / float(len(precisionlist_1)))
    print("Recall - Class 0", sum(recalllist0) / float(len(recalllist0)))
    print("Recall - Class 1", sum(recalllist1) / float(len(recalllist1)))
    print("Recall - Class -1", sum(recalllist_1) / float(len(recalllist_1)))
    print("F1 Score - Class 0", sum(f1scorelist0) / float(len(f1scorelist0)))
    print("F1 Score - Class 1", sum(f1scorelist1) / float(len(f1scorelist1)))
    print("F1 Score - Class -1", sum(f1scorelist_1) / float(len(f1scorelist_1)))
    print("Overall Accuracy ", sum(accuracylist) / float(len(accuracylist)))

calculate_metrics = CalculateMetrics()
tweets, tweetlist, labels = calculate_metrics.read_file("Obama_data_cleaned.txt")
#word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))

number_of_folds = 10
subset_size = len(tweetlist)/number_of_folds
mnb_precisionlist0 = []
mnb_precisionlist1 = []
mnb_precisionlist_1=[]
mnb_recalllist0 =[]
mnb_recalllist1=[]
mnb_recalllist_1=[]
mnb_f1scorelist0=[]
mnb_f1scorelist1=[]
mnb_f1scorelist_1=[]
mnb_accuracylist = []
gnb_precisionlist0=[]
gnb_precisionlist1=[]
gnb_precisionlist_1=[]
gnb_recalllist0=[]
gnb_recalllist1=[]
gnb_recalllist_1=[]
gnb_f1scorelist0=[]
gnb_f1scorelist1=[]
gnb_f1scorelist_1=[]
gnb_accuracylist = []
bnb_precisionlist0=[]
bnb_precisionlist1=[]
bnb_precisionlist_1=[]
bnb_recalllist0=[]
bnb_recalllist1=[]
bnb_recalllist_1=[]
bnb_f1scorelist0=[]
bnb_f1scorelist1=[]
bnb_f1scorelist_1=[]
bnb_accuracylist = []
knn_precisionlist0=[]
knn_precisionlist1=[]
knn_precisionlist_1=[]
knn_recalllist0=[]
knn_recalllist1=[]
knn_recalllist_1=[]
knn_f1scorelist0=[]
knn_f1scorelist1=[]
knn_f1scorelist_1=[]
knn_accuracylist = []
dt_precisionlist0=[]
dt_precisionlist1=[]
dt_precisionlist_1=[]
dt_recalllist0=[]
dt_recalllist1=[]
dt_recalllist_1=[]
dt_f1scorelist0=[]
dt_f1scorelist1=[]
dt_f1scorelist_1=[]
dt_accuracylist = []
ab_precisionlist0=[]
ab_precisionlist1=[]
ab_precisionlist_1=[]
ab_recalllist0=[]
ab_recalllist1=[]
ab_recalllist_1=[]
ab_f1scorelist0=[]
ab_f1scorelist1=[]
ab_f1scorelist_1=[]
ab_accuracylist = []
rf_precisionlist0=[]
rf_precisionlist1=[]
rf_precisionlist_1=[]
rf_recalllist0=[]
rf_recalllist1=[]
rf_recalllist_1=[]
rf_f1scorelist0=[]
rf_f1scorelist1=[]
rf_f1scorelist_1=[]
rf_accuracylist = []


for i in range(number_of_folds):
    test_data = tweetlist[i*subset_size:][:subset_size]
    train_data = tweetlist[:i*subset_size] + tweetlist[(i+1)*subset_size:]
    test_data_labels = labels[i*subset_size:][:subset_size]
    train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]
    tfidf_vectorizer = TfidfVectorizer()
#    tfidf_vectorizer = CountVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data)

#    cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
    mnb_classifier = MultinomialNB().fit(tfidf_matrix, train_data_labels)
    predictions = mnb_classifier.predict(test_tfidf_matrix)
    mnb_precisionlist0, mnb_precisionlist1, mnb_precisionlist_1, mnb_recalllist0, mnb_recalllist1, mnb_recalllist_1, mnb_f1scorelist0, mnb_f1scorelist1, mnb_f1scorelist_1, mnb_accuracylist = evaluation_metrics(mnb_precisionlist0, mnb_precisionlist1, mnb_precisionlist_1, mnb_recalllist0, mnb_recalllist1, mnb_recalllist_1, mnb_f1scorelist0, mnb_f1scorelist1, mnb_f1scorelist_1, mnb_accuracylist)
    
    gnb_classifier = GaussianNB().fit(tfidf_matrix, train_data_labels)
    predictions = gnb_classifier.predict(test_tfidf_matrix)
    gnb_precisionlist0, gnb_precisionlist1, gnb_precisionlist_1, gnb_recalllist0, gnb_recalllist1, gnb_recalllist_1, gnb_f1scorelist0, gnb_f1scorelist1, gnb_f1scorelist_1, gnb_accuracylist = evaluation_metrics(gnb_precisionlist0, gnb_precisionlist1, gnb_precisionlist_1, gnb_recalllist0, gnb_recalllist1, gnb_recalllist_1, gnb_f1scorelist0, gnb_f1scorelist1, gnb_f1scorelist_1, gnb_accuracylist)
    
    bnb_classifier = BernoulliNB().fit(tfidf_matrix, train_data_labels)
    predictions = bnb_classifier.predict(test_tfidf_matrix)
    bnb_precisionlist0, bnb_precisionlist1, bnb_precisionlist_1, bnb_recalllist0, bnb_recalllist1, bnb_recalllist_1, bnb_f1scorelist0, bnb_f1scorelist1, bnb_f1scorelist_1, bnb_accuracylist = evaluation_metrics(bnb_precisionlist0, bnb_precisionlist1, bnb_precisionlist_1, bnb_recalllist0, bnb_recalllist1, bnb_recalllist_1, bnb_f1scorelist0, bnb_f1scorelist1, bnb_f1scorelist_1, bnb_accuracylist)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=3).fit(tfidf_matrix, train_data_labels)
    predictions = knn_classifier.predict(test_tfidf_matrix)
    knn_precisionlist0, knn_precisionlist1, knn_precisionlist_1, knn_recalllist0, knn_recalllist1, knn_recalllist_1, knn_f1scorelist0, knn_f1scorelist1, knn_f1scorelist_1, knn_accuracylist = evaluation_metrics(knn_precisionlist0, knn_precisionlist1, knn_precisionlist_1, knn_recalllist0, knn_recalllist1, knn_recalllist_1, knn_f1scorelist0, knn_f1scorelist1, knn_f1scorelist_1, knn_accuracylist)
    
    dt_classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = dt_classifier.predict(test_tfidf_matrix)
    dt_precisionlist0, dt_precisionlist1, dt_precisionlist_1, dt_recalllist0, dt_recalllist1, dt_recalllist_1, dt_f1scorelist0, dt_f1scorelist1, dt_f1scorelist_1, dt_accuracylist = evaluation_metrics(dt_precisionlist0, dt_precisionlist1, dt_precisionlist_1, dt_recalllist0, dt_recalllist1, dt_recalllist_1, dt_f1scorelist0, dt_f1scorelist1, dt_f1scorelist_1, dt_accuracylist)
    
    ab_classifier = AdaBoostClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = ab_classifier.predict(test_tfidf_matrix)
    ab_precisionlist0, ab_precisionlist1, ab_precisionlist_1, ab_recalllist0, ab_recalllist1, ab_recalllist_1, ab_f1scorelist0, ab_f1scorelist1, ab_f1scorelist_1, ab_accuracylist = evaluation_metrics(ab_precisionlist0, ab_precisionlist1, ab_precisionlist_1, ab_recalllist0, ab_recalllist1, ab_recalllist_1, ab_f1scorelist0, ab_f1scorelist1, ab_f1scorelist_1, ab_accuracylist)
    
    rf_classifier = RandomForestClassifier().fit(tfidf_matrix, train_data_labels)
    predictions = rf_classifier.predict(test_tfidf_matrix)
    rf_precisionlist0, rf_precisionlist1, rf_precisionlist_1, rf_recalllist0, rf_recalllist1, rf_recalllist_1, rf_f1scorelist0, rf_f1scorelist1, rf_f1scorelist_1, rf_accuracylist = evaluation_metrics(rf_precisionlist0, rf_precisionlist1, rf_precisionlist_1, rf_recalllist0, rf_recalllist1, rf_recalllist_1, rf_f1scorelist0, rf_f1scorelist1, rf_f1scorelist_1, rf_accuracylist)
    
print("Multinomial Naive Bayes Classifier - Obama")
print_values(mnb_precisionlist0, mnb_precisionlist1, mnb_precisionlist_1, mnb_recalllist0, mnb_recalllist1, mnb_recalllist_1, mnb_f1scorelist0, mnb_f1scorelist1, mnb_f1scorelist_1, mnb_accuracylist)

print("Gaussian Naive Bayes Classifier - Obama")
print_values(gnb_precisionlist0, gnb_precisionlist1, gnb_precisionlist_1, gnb_recalllist0, gnb_recalllist1, gnb_recalllist_1, gnb_f1scorelist0, gnb_f1scorelist1, gnb_f1scorelist_1, gnb_accuracylist)

print("Bernoulli Naive Bayes Classifier - Obama")
print_values(bnb_precisionlist0, bnb_precisionlist1, bnb_precisionlist_1, bnb_recalllist0, bnb_recalllist1, bnb_recalllist_1, bnb_f1scorelist0, bnb_f1scorelist1, bnb_f1scorelist_1, bnb_accuracylist)

print("K Nearest Neighbors Classifier - Obama")
print_values(knn_precisionlist0, knn_precisionlist1, knn_precisionlist_1, knn_recalllist0, knn_recalllist1, knn_recalllist_1, knn_f1scorelist0, knn_f1scorelist1, knn_f1scorelist_1, knn_accuracylist)

print("Decision Trees Classifier - Obama")
print_values(dt_precisionlist0, dt_precisionlist1, dt_precisionlist_1, dt_recalllist0, dt_recalllist1, dt_recalllist_1, dt_f1scorelist0, dt_f1scorelist1, dt_f1scorelist_1, dt_accuracylist)

print("AdaBoost Classifier - Obama")
print_values(ab_precisionlist0, ab_precisionlist1, ab_precisionlist_1, ab_recalllist0, ab_recalllist1, ab_recalllist_1, ab_f1scorelist0, ab_f1scorelist1, ab_f1scorelist_1, ab_accuracylist)

print("Random Forest Classifier - Obama")
print_values(rf_precisionlist0, rf_precisionlist1, rf_precisionlist_1, rf_recalllist0, rf_recalllist1, rf_recalllist_1, rf_f1scorelist0, rf_f1scorelist1, rf_f1scorelist_1, rf_accuracylist)