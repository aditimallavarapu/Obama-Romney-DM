# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:39:27 2017

@author: Suganya
"""
import os
import nltk
from random import shuffle
from preprocess import Preprocess
from textblob import TextBlob as tb

class TwitterSentimentAnalysis:
    def get_words_in_tweets(self, text):
        all_words = []
        for (words, sentiment) in text:
          all_words.extend(words)
        return all_words
    
    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        hapaxes = wordlist.hapaxes()           
        features_final= [word for word in wordlist if word not in hapaxes]
        #print features_final
        features_short = features_final[0:10000]
        return features_short
        
    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in word_features:
            #features['contains(%s)' % word] = (word in document_words)
            features[word] = (word in document_words)
        return features
        
    def generate_ngrams(self, num, tweet):
        tweet_tokens = nltk.word_tokenize(tweet)
        ngram = []
        for i, word in enumerate(tweet_tokens):
            for n in range(1, num + 1):
                if i + n <= len(tweet_tokens):
                    ngram_list = [tweet_tokens[j] for j in xrange(i, i + n)]   
                    if(len(ngram_list) == num):
                        ngram.append(' '.join(ngram_list).lower())
        #print(ngram)
        return ngram
        
    def generate_input_tokens(self, num, tweet):
        tweet_tokens = nltk.word_tokenize(tweet)
        ngram = []
        for i in enumerate(tweet_tokens):
            for n in range(1, num + 1):
                if i + n <= len(tweet_tokens):
                    ngram_list = [tweet_tokens[j] for j in xrange(i, i + n)]   
                    ngram.append(' '.join(ngram_list).lower())
        #print(ngram)
        return ngram
        
    def read_file(self, filename,num_gram):
        rel_path = filename
        script_dir= os.path.dirname(os.path.abspath(__file__))
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open ( abs_file_path )
        tweetlist = []
        labels = []
        tweets=[]
        preprocess = Preprocess()
   
        for line in f.readlines():
            cols = line.split("\t")
            cols[0] = preprocess.cleanup(cols[0])      #write to a file new cleaned things 
            if(num_gram==1):          
                words_filtered=[]   #remove words less than 2 letters in length
                words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
                tweets.append((words_filtered,cols[1]))
            elif(num_gram==2): 
               bigrams_list = self.generate_ngrams(2, cols[0])
               if(len(bigrams_list) > 0):
                   tweets.append((bigrams_list,cols[1]))
            elif(num_gram==3):  
               trigrams_list = self.generate_ngrams(3, cols[0])
               if(len(trigrams_list) > 0):
                   tweets.append((trigrams_list,cols[1]))
            elif(num_gram==4):
               quadgrams_list = self.generate_ngrams(4, cols[0])
               if(len(quadgrams_list) > 0):
                    tweets.append((quadgrams_list,cols[1]))
            tweetlist.append(cols[0])
            labels.append(cols[1])
        f.close()
        shuffle(tweets)
        return tweets, tweetlist, labels
        
    """
    To compare labels from test set vs predicted labels
    """
    def calculate_metrics(self,classifier, test_set, class_label):
        #precision, recall, F1score
        results = classifier.classify_many([fs for (fs, l) in test_set])
        tp = [l == class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        fp = [l != class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        #tn = [l != class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        fn = [l == class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        classified_correct = [l == r for ((fs, l), r) in zip(test_set, results)]
        precision, recall, f1score, overall_accuracy = self.calculate_values(classified_correct, tp, fp, fn)
        return precision, recall, f1score, overall_accuracy
        
    def calculate_values(self, classified_correct, tp, fp, fn):
        precision_denominator = (sum(tp) + sum(fp))
        recall_denominator = (sum(tp) + sum(fn))
        if(precision_denominator != 0.0):
            precision = float(sum(tp)) / precision_denominator
        else:
            #precision = 'NAN'
            precision = 0.0
        if(recall_denominator != 0.0):
            recall = float(sum(tp)) / (sum(tp) + sum(fn))
        else:
            #recall = 'NAN'
            recall = 0.0
        if(precision+recall != 0.0):
            f1score = float(2*precision*recall) / float(precision + recall)
        else:
            f1score = 'NAN'
        if classified_correct:
            overall_accuracy = float(sum(classified_correct)) / len(classified_correct)
        else:
            overall_accuracy = 0.0
        return precision, recall, f1score, overall_accuracy
        
    """
    To compare predicted labels vs actual labels
    """
    def metrics(self, class_label, predicted_labels, actual_labels):
        assert(len(predicted_labels) == len(actual_labels))
        tp = [l == class_label and r == class_label for (l, r) in zip(actual_labels, predicted_labels)]
        fp = [l != class_label and r == class_label for (l, r) in zip(actual_labels, predicted_labels)]
        fn = [l == class_label and r != class_label for (l, r) in zip(actual_labels, predicted_labels)]
        n =  [l == class_label for l in actual_labels]
        class_accuracy = float(sum(tp)) / sum(n)
        overall_classified_correct = [l == r for (l,r) in zip(actual_labels, predicted_labels)]
        precision, recall, f1score, overall_accuracy = self.calculate_values(overall_classified_correct, tp, fp, fn)
        return precision, recall, f1score, class_accuracy, overall_accuracy