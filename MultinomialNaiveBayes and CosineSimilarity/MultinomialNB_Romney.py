# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:42:46 2017

@author: Suganya
"""

from TwitterSentimentAnalysis import TwitterSentimentAnalysis
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB


analysis = TwitterSentimentAnalysis()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt",1)
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))

test_tweets, test_tweetlist, test_labels = analysis.read_file("Romney_test_data_cleaned.txt",1)

number_of_folds = 10
subset_size = len(tweetlist)/number_of_folds
f1scorelist0 = []
f1scorelist1 = []
f1scorelist_1 = []
accuracylist = []

for i in range(number_of_folds):
    test_data = tweetlist[i*subset_size:][:subset_size]
    train_data = tweetlist[:i*subset_size] + tweetlist[(i+1)*subset_size:]
    test_data_labels = labels[i*subset_size:][:subset_size]
    train_data_labels = labels[:i*subset_size] + labels[(i+1)*subset_size:]
    #tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = CountVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data)

    cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
    clf = MultinomialNB().fit(tfidf_matrix, train_data_labels)

    predictions = clf.predict(test_tfidf_matrix)
    precision0, recall0, f1score0, class_accuracy0, overall_accuracy = analysis.metrics("0\n", predictions, test_data_labels)
    precision1, recall1, f1score1, class_accuracy1, overall_accuracy = analysis.metrics("1\n", predictions, test_data_labels)
    precision_1, recall_1, f1score_1, class_accuracy_1, overall_accuracy = analysis.metrics("-1\n", predictions, test_data_labels)
    f1scorelist0.append(f1score0)
    f1scorelist1.append(f1score1)
    f1scorelist_1.append(f1score_1)
    accuracylist.append((class_accuracy0 + class_accuracy1 + class_accuracy_1) / 3)
print("F1 Score - Class 0", sum(f1scorelist0) / float(len(f1scorelist0)))
print("F1 Score - Class 1", sum(f1scorelist1) / float(len(f1scorelist1)))
print("F1 Score - Class -1", sum(f1scorelist_1) / float(len(f1scorelist_1)))
print("Accuracy ", sum(accuracylist) / float(len(accuracylist)))



