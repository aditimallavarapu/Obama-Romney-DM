# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:26:43 2017

@author: Aditi
"""
import os
import re
import string
import nltk


def cleanup(data):
    cleantext = data.replace(",","")        #replace commas
    cleaner = re.compile('<.*?>')           #replace tags with space
    cleantext= re.sub(cleaner,'', cleantext)        
    ascii = set(string.printable) 
    cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
    cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
    cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
    #may want to remove numbers
    return cleantext
         
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
    

   

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
"""Read actual file have file name here"""
rel_path = "training_obama_tweets_nodate.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f = open(abs_file_path, "r")
lines = f.read().split("\n")
num_lines = sum(1 for line in open(abs_file_path))
tweets=[]
#fout=open(abs_file_path+"1", 'w+')   write into this file for weka
for i in range(0,num_lines):
    cols = lines[i].split("\t")
    cols[0] = cleanup(cols[0])      #write to a file new cleaned things 
    #unigrams
    words_filtered=[]   #remove words less than 2 letters in length
    words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
    tweets.append((words_filtered,cols[1]))
    #bigram
    #trigram
    #quadgram

    
f.close()
#fout.close()
word_features = get_word_features(get_words_in_tweets(tweets))
def extract_features(document):
    #print document
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
training_set = nltk.classify.apply_features(extract_features, tweets)
#print training_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
#print classifier.show_most_informative_features(32)

tweet = 'China hands Leads'
print classifier.classify(extract_features(tweet.split()))

