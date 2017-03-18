# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:14:39 2017

@authors: Aditi and Suganya
"""

import os
import re
import string
import nltk
from nltk.corpus import stopwords
from random import shuffle


class TwitterSentimentAnalysis:
    def cleanup(self, data):
        cleantext = data.replace(",","")        #remove commas
        cleaner = re.compile('<.*?>')           #remove tags
        cleantext= re.sub(cleaner,'', cleantext)        
        ascii = set(string.printable) 
        cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
        cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
        cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
        cleantext = cleantext.translate(None, string.digits)
        stop = set(stopwords.words('english')) - set(('and', 'or', 'not'))
        cleantextlist = [i for i in cleantext.lower().split() if i not in stop]      #remove stopwords except few exceptions  
        cleantext = ' '.join(cleantextlist)
        
        #cleantext = re.sub(".*\d+.*", " ", cleantext)     #replace remove numbers
        return cleantext
             
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
        tweets=[]
   
        for line in f.readlines():
            cols = line.split("\t")
            cols[0] = self.cleanup(cols[0])      #write to a file new cleaned things 
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
        f.close()
        shuffle(tweets)
        return tweets
        
    def calculate_metrics(self,classifier, test_set, class_label):
        #precision, recall, F1score
        results = classifier.classify_many([fs for (fs, l) in test_set])
        tp = [l == class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        fp = [l != class_label and r == class_label for ((fs, l), r) in zip(test_set, results)]
        #tn = [l != class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        fn = [l == class_label and r != class_label for ((fs, l), r) in zip(test_set, results)]
        classified_correct = [l == r for ((fs, l), r) in zip(test_set, results)]
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
        
romney_model = TwitterSentimentAnalysis()
tweets = romney_model.read_file("test_romney.txt",1)
word_features = romney_model.get_word_features(romney_model.get_words_in_tweets(tweets))   
