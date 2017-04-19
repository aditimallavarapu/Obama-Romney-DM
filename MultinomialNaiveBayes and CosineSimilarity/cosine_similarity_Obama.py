# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:42:46 2017

@author: Suganya
"""

from TwitterSentimentAnalysis import TwitterSentimentAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mode
        

analysis = TwitterSentimentAnalysis()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt",1)
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))

test_tweets, test_tweetlist, test_labels = analysis.read_file("Obama_test_data_cleaned.txt",1)

tfidf_vectorizer = TfidfVectorizer()
#tfidf_vectorizer = CountVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
test_tfidf_matrix = tfidf_vectorizer.transform(test_tweetlist)

cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
predicted_labels = []
top_predictions = []

#classification based on cosine_similarity
for i in xrange(0, len(test_labels)):
    related_tweet_indices = sorted(range(len(cosine_similarities[i])), key=lambda j: cosine_similarities[i][j])[-3:]
    top_predictions = [labels[related_tweet_indices[k]] for k in xrange(0,3)]
    predicted_labels.append(mode(top_predictions).mode[0])
    
precision0, recall0, f1score0, class_accuracy0, overall_accuracy = analysis.metrics("0\n", predicted_labels, test_labels)
precision1, recall1, f1score1, class_accuracy1, overall_accuracy = analysis.metrics("1\n", predicted_labels, test_labels)
precision_1, recall_1, f1score_1, class_accuracy_1, overall_accuracy = analysis.metrics("-1\n", predicted_labels, test_labels)
print("F1 Score - Class 0", f1score0)
print("F1 Score - Class 1", f1score1)
print("F1 Score - Class -1", f1score_1)
print("Accuracy ", (class_accuracy0 + class_accuracy1 + class_accuracy_1) / 3)




