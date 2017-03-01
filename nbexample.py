# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:26:43 2017

@author: Aditi
"""
import os
import re
import string
import nltk

class TwitterSentimentAnalysis:
    def cleanup(self, data):
        cleantext = data.replace(",","")        #remove commas
        cleaner = re.compile('<.*?>')           #remove tags
        cleantext= re.sub(cleaner,'', cleantext)        
        ascii = set(string.printable) 
        cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
        cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
        cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
        #may want to remove numbers
        return cleantext
             
    def get_words_in_tweets(self, tweets):
        all_words = []
        for (words, sentiment) in tweets:
          all_words.extend(words)
        return all_words
    
    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features = wordlist.keys()
        return word_features
        
    def extract_features(self, document):
        #print document
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
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

   
analysis = TwitterSentimentAnalysis()
script_dir = os.path.dirname("") #<-- absolute dir the script is in
"""Read actual file have file name here"""
rel_path = "training_obama_tweets_nodate_small.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f = open ( rel_path )
tweets=[]
#fout=open(abs_file_path+"1", 'w+')   write into this file for weka
for line in f.readlines():
    cols = line.split("\t")
    cols[0] = analysis.cleanup(cols[0])      #write to a file new cleaned things 
    #unigrams
    words_filtered=[]   #remove words less than 2 letters in length
    words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
    tweets.append((words_filtered,cols[1]))
    
    #bigram
    bigrams_list = analysis.generate_ngrams(2, cols[0])
    if(len(bigrams_list) > 0):
        tweets.append((bigrams_list,cols[1]))
    
    #trigram
    trigrams_list = analysis.generate_ngrams(3, cols[0])
    if(len(trigrams_list) > 0):
        tweets.append((trigrams_list,cols[1]))
    #quadgram
    quadgrams_list = analysis.generate_ngrams(4, cols[0])
    if(len(quadgrams_list) > 0):
        tweets.append((quadgrams_list,cols[1]))
    """
    """
    
f.close()
#fout.close()
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))

    
training_set = nltk.classify.apply_features(analysis.extract_features, tweets)
#print training_set
print("Starting...")
classifier = nltk.NaiveBayesClassifier.train(training_set)
#print classifier.show_most_informative_features(32)
print("Model built...")
tweet = 'China hands Leads'
print classifier.classify(analysis.extract_features(tweet.split())) #we have to modify this to use the ngram model as well

