# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:42:46 2017

@author: Suganya
"""

from TwitterSentimentAnalysis import TwitterSentimentAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
        

analysis = TwitterSentimentAnalysis()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt",1)
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))

test_tweets, test_tweetlist, test_labels = analysis.read_file("Obama_test_data_cleaned.txt",1)

number_of_folds = 10
subset_size = len(tweetlist)/number_of_folds
precisionlist0 = []
precisionlist1 = []
precisionlist_1 = []
recalllist0 = []
recalllist1 = []
recalllist_1 = []
f1scorelist0 = []
f1scorelist1 = []
f1scorelist_1 = []
accuracylist = []

for i in range(number_of_folds):
    test_data = tweetlist[i*subset_size:][:subset_size]
    train_data = tweetlist[:i*subset_size] + tweetlist[(i+1)*subset_size:]
    test_data_labels = labels[i*subset_size:][:subset_size]
    train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]
    tfidf_vectorizer = TfidfVectorizer()
#    tfidf_vectorizer = CountVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data)

    cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
    classifier = GaussianNB().fit(tfidf_matrix.toarray(), train_data_labels)

    predictions = classifier.predict(test_tfidf_matrix.toarray())
    
    precision0, recall0, f1score0, class_accuracy0, overall_accuracy = analysis.metrics("0\n", predictions, test_data_labels)
    precision1, recall1, f1score1, class_accuracy1, overall_accuracy = analysis.metrics("1\n", predictions, test_data_labels)
    precision_1, recall_1, f1score_1, class_accuracy_1, overall_accuracy = analysis.metrics("-1\n", predictions, test_data_labels)
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
print("Gaussian Naive Bayes Classifier - Obama")
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
