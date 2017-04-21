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
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

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
classifier = MultinomialNB(alpha=0.8)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('MNB', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Multinomial_Obama_classifier.pickle', compress=9)
                             
"""Bernouli Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=8,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
classifier = BernoulliNB(alpha=1.58)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('BNB', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Bernouli_Obama_classifier.pickle', compress=9)


"""Desicion tree Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1), min_df=15,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
classifier = tree.DecisionTreeClassifier()      
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('DT', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Decision_tree_Obama_classifier.pickle', compress=9)
                       

"""Adaboost Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=15,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
classifier = AdaBoostClassifier(n_estimators=58)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('AB', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Adaboost_Obama_classifier.pickle', compress=9)

"""KNN Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,4), min_df=30,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
classifier = KNeighborsClassifier(n_neighbors=7)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('KNN', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'KNN_Obama_classifier.pickle', compress=9)

"""Random Forests Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
classifier = RandomForestClassifier(n_estimators=27)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('RF', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Random_Forests_Obama_classifier.pickle', compress=9)

"""Linear SVM Obama Model""" 
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
classifier = svm.LinearSVC(C=0.20)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'LinearSVM_Obama_classifier.pickle', compress=9)

"""Polynomial SVM Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = svm.SVC(kernel = 'poly',degree=2)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm-poly', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Polynomial_SVM_Obama_classifier.pickle', compress=9)

"""RBF SVM Obama Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                             max_df = 1.0,
                             sublinear_tf=True,
                             use_idf=True)
classifier = svm.SVC(kernel = 'rbf',gamma=0.81)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm-RBF', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'RBF_SVM_Obama_classifier.pickle', compress=9)

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
classifier = MultinomialNB(alpha=0)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('MNB', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Multinomial_Romney_classifier.pickle', compress=9)
 
#"""cosine Obama Model"""
#analysis= CalculateMetrics()
#tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
#tfidf_vectorizer = TfidfVectorizer()
#tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
#
#f = open('cosine_Obama_classifier.pickle', 'wb')
#pickle.dump(tfidf_matrix,f)
#f.close()
# 
#"""cosine Romney Model"""
#analysis= CalculateMetrics()
#tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
#tfidf_vectorizer = TfidfVectorizer()
#tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
##cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
#f = open('cosine_Romney_classifier.pickle', 'wb')
#pickle.dump(tfidf_matrix,f)
#f.close()
# 
 
"""Adaboost Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=15,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)

classifier = AdaBoostClassifier(n_estimators=57)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('Ab', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Adaboost_Romney_classifier.pickle', compress=9)
 
"""Bernouli Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=7,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
classifier = BernoulliNB(alpha=1.06)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('BNB', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Bernouli_Romney_classifier.pickle', compress=9)

"""Decision tree Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=15,
                              max_df = 0.9,
                              sublinear_tf=True,
                              use_idf=True)
classifier = tree.DecisionTreeClassifier()
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('DT', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Decision_tree_Romney_classifier.pickle', compress=9)
 
"""KNN Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1), min_df=31,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = KNeighborsClassifier(n_neighbors=9)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('KNN', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'KNN_Romney_classifier.pickle', compress=9)

"""Random Forests Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = RandomForestClassifier(n_estimators=35)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('RF', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Random_Forests_Romney_classifier.pickle', compress=9)

"""Linear SVM Romney Model""" 
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = svm.LinearSVC(C=0.35)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'LinearSVM_Romney_classifier.pickle', compress=9)

"""Polynomial SVM Romeny Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df=0,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = svm.SVC(kernel = 'poly',degree=6)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm-Poly', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'Polynomial_SVM_Romney_classifier.pickle', compress=9)

"""RBF SVM Romney Model"""
analysis= CalculateMetrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3), min_df=5,
                              max_df = 1.0,
                              sublinear_tf=True,
                              use_idf=True)
classifier = svm.SVC(kernel = 'rbf',gamma=0.75)
vec_clf = Pipeline([('tfvec', tfidf_vectorizer), ('svm-RBF', classifier)])
vec_clf.fit(tweetlist, labels)
joblib.dump(vec_clf, 'RBF_SVM_Romney_classifier.pickle', compress=9)
