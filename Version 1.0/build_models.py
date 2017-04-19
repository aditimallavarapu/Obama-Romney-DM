# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:32:10 2017

@author: Aditi
"""
from CalculateMetrics import CalculateMetrics
from preprocess import Preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from Trainmodel import Naive_Bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nltk
import pickle
import os
from sklearn import svm

"""preprocess all training files"""
process = Preprocess() 
script_dir = os.path.dirname("") #<-- absolute dir the script is in
process.xls_to_txt('training-Obama-Romney-tweets-Recoded.xlsx','Obama_data.txt','Romney_data.txt')
print("Text file saved")           
process.clean_text_files('Romney_data_cleaned.txt','Romney_data.txt','Obama_data_cleaned.txt','Obama_data.txt')
process.remove_2("Obama_data_cleaned.txt")
process.remove_2("Romney_data_cleaned.txt")

#Obama:

"""Unigram Obama Model"""
analysis = Naive_Bayes("Obama_data_cleaned.txt",1)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Unigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('unigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bigram Obama Model"""
analysis = Naive_Bayes("Obama_data_cleaned.txt",2)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Bigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Bigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Trigram Obama Model"""
analysis = Naive_Bayes("Obama_data_cleaned.txt",3)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...trigram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('trigram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Multinomial Obama Model"""
analysis=CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=8,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = MultinomialNB(alpha=0.8).fit(tfidf_matrix, labels)
f = open('Multinomial_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bernouli Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=8,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = BernoulliNB(alpha=1.58).fit(tfidf_matrix, labels)
f = open('Bernouli_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Desicion tree Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1), min_df=15,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, labels)
f = open('Decision_tree_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Adaboost Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=15,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = AdaBoostClassifier(n_estimators=58).fit(tfidf_matrix, labels)
f = open('Adaboost_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""KNN Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,4), min_df=30,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = KNeighborsClassifier(n_neighbors=7).fit(tfidf_matrix, labels)
f = open('KNN_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Random Forests Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = RandomForestClassifier(n_estimators=27).fit(tfidf_matrix, labels)
f = open('Random_Forests_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Linear SVM Obama Model""" 
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.LinearSVC(C=0.20).fit(tfidf_matrix, labels)
f = open('LinearSVM_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Polynomial SVM Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.SVC(kernel = 'poly',degree=2).fit(tfidf_matrix, labels)
f = open('Polynomial_SVM_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close() 

"""RBF SVM Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.SVC(kernel = 'rbf',gamma=0.81).fit(tfidf_matrix, labels)
f = open('RBF_SVM_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close() 




#Romney:
"""Unigram Romney Model"""
analysis = Naive_Bayes("Romney_data_cleaned.txt",1)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Unigram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('unigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
 
"""Bigram Romney Model"""
analysis = Naive_Bayes("Romney_data_cleaned.txt",2)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...Bigram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Bigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
 
"""Trigram Romney Model"""
analysis = Naive_Bayes("Romney_data_cleaned.txt",3)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('trigram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Multinomial Romney Model"""
analysis=CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=6,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = MultinomialNB(alpha=0).fit(tfidf_matrix, labels)
f = open('Multinomial_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
 
"""cosine Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
#cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
f = open('cosine_Obama_classifier.pickle', 'wb')
pickle.dump(tfidf_matrix,f)
f.close()
 
"""cosine Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
#cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
f = open('cosine_Romney_classifier.pickle', 'wb')
pickle.dump(tfidf_matrix,f)
f.close()
 
 
"""Adaboost Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=15,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = AdaBoostClassifier(n_estimators=57).fit(tfidf_matrix, labels)
f = open('Adaboost_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
 
"""Bernouli Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=7,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = BernoulliNB(alpha=1.06).fit(tfidf_matrix, labels)
f = open('Bernouli_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Decision tree Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=15,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, labels)
f = open('Decision_tree_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
 
"""KNN Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1), min_df=31,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = KNeighborsClassifier(n_neighbors=9).fit(tfidf_matrix, labels)
f = open('KNN_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()


"""Random Forests Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = RandomForestClassifier(n_estimators=35).fit(tfidf_matrix, labels)
f = open('Random_Forests_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()


"""Linear SVM Romney Model""" 
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.LinearSVC(C=0.35).fit(tfidf_matrix, labels)
f = open('LinearSVM_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Polynomial SVM Romeny Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.SVC(kernel = 'poly',degree=6).fit(tfidf_matrix, labels)
f = open('Polynomial_SVM_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close() 

"""RBF SVM Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = svm.SVC(kernel = 'rbf',gamma=0.75).fit(tfidf_matrix, labels)
f = open('RBF_SVM_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close() 
