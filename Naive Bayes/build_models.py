# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:32:10 2017

@author: Aditi
"""
from Calculate_metrics import Calculate_metrics
from preprocess import Preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from Trainmodel import Naive_Bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nltk
import pickle

"""preprocess all training files"""
process = Preprocess() 
#script_dir = os.path.dirname("") #<-- absolute dir the script is in
process.xls_to_txt('training-Obama-Romney-tweets.xlsx','Obama_data.txt','Romney_data.txt')
print("Text file saved")           
process.clean_text_files('Romney_data_cleaned.txt','Romney_data.txt','Obama_data_cleaned.txt','Obama_data.txt')

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

"""Quadgram Obama Model"""
analysis = Naive_Bayes("Obama_data_cleaned.txt",4)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...quadgram modelling for Obama")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Quadgram_Obama_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

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

"""Quadgram Romney Model"""
analysis = Naive_Bayes("Obama_data_cleaned.txt",4)
training_set = nltk.classify.apply_features(analysis.extract_features, analysis.tweets)
print("Starting...quadgram modelling for Romney")
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('Quadgram_Romney_nb_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Multinomial Obama Model"""
analysis=Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = CountVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = MultinomialNB().fit(tfidf_matrix, labels)
f = open('Multinomial_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()
#add this to the test_model file
#test_tweets, test_tweetlist, test_labels = analysis.read_file("Obama_test_data_cleaned.txt")

"""Multinomial Romney Model"""
analysis=Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = CountVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = MultinomialNB().fit(tfidf_matrix, labels)
f = open('Multinomial_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""cosine Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
#cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
f = open('cosine_Obama_classifier.pickle', 'wb')
pickle.dump(tfidf_matrix,f)
f.close()

"""cosine Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
#cosine_similarities = cosine_similarity(test_tfidf_matrix, tfidf_matrix)
f = open('cosine_Romney_classifier.pickle', 'wb')
pickle.dump(tfidf_matrix,f)
f.close()

"""Adaboost Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = AdaBoostClassifier().fit(tfidf_matrix, labels)
f = open('Adaboost_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Adaboost Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = AdaBoostClassifier().fit(tfidf_matrix, labels)
f = open('Adaboost_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bernouli Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = BernoulliNB().fit(tfidf_matrix, labels)
f = open('Bernouli_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Bernouli Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = BernoulliNB().fit(tfidf_matrix, labels)
f = open('Bernouli_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Desicion tree Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, labels)
f = open('Decision_tree_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Decision tree Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = tree.DecisionTreeClassifier().fit(tfidf_matrix, labels)
f = open('Decision_tree_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()


"""Gaussian NB Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = GaussianNB().fit(tfidf_matrix.toarray(), labels)
f = open('GaussianNB_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Gaussian NB Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = GaussianNB().fit(tfidf_matrix.toarray(), labels)
f = open('GaussianNB_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""KNN Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = KNeighborsClassifier(n_neighbors=3).fit(tfidf_matrix, labels)
f = open('KNN_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""KNN Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = KNeighborsClassifier(n_neighbors=3).fit(tfidf_matrix, labels)
f = open('KNN_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()


"""Random Forests Obama Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Obama_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = RandomForestClassifier().fit(tfidf_matrix, labels)
f = open('Random_Forests_Obama_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

"""Random Forests Romney Model"""
analysis= Calculate_metrics()
tweets, tweetlist, labels = analysis.read_file("Romney_data_cleaned.txt")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tweetlist)
classifier = RandomForestClassifier().fit(tfidf_matrix, labels)
f = open('Random_Forests_Romney_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

