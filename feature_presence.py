# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 18:03:00 2017

@author: Aditi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:26:43 2017

@author: Aditi & Suganya
"""
import os
import re
import string
import nltk
import collections

class TwitterSentimentAnalysis:
    def cleanup(self, data):
        cleantext = data.replace(",","")        #replace commas
        cleaner = re.compile('<.*?>')           #replace tags
        cleantext= re.sub(cleaner,'', cleantext)        
        ascii = set(string.printable) 
        cleantext=filter(lambda x: x in ascii , cleantext)   #remove emoji and non english characters
        cleantext= re.sub(r'https?:\/\/.*[\r\n]*', '', cleantext)       #remove links
        cleantext = cleantext.translate(string.maketrans("",""), string.punctuation)
        cleantext = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", cleantext)     #replace remove numbers
        return cleantext
             
    def get_words_in_tweets(self, tweets):
        all_words = []
        for (words, sentiment) in tweets:
          all_words.extend(words)
        return all_words
    
    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        word_features=wordlist.keys()
        return word_features
        
   
        
       
        
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

def extract_features(self,document):
        """document has current line
        need to find what words occur in this line """
       # print document
        # print('Hello', s, '!')
        document_words = set(document)
        features = {}     #need to initialise to 0 for all features in word features
        for word in word_features:      #instead of contains should i create a frequency map
            features['contains(%s)' % word] = (word in document_words)
            
        return features
        
        
analysis = TwitterSentimentAnalysis()
script_dir = os.path.dirname("") #<-- absolute dir the script is in
"""Read actual file have file name here"""
rel_path = "training_obama_tweets_nodate_small.txt"   #merge both files
abs_file_path = os.path.join(script_dir, rel_path)
f = open ( rel_path )
lines = f.read().split("\n")
f.close();
tweets=[]
num_lines= sum(1 for line in open(abs_file_path))
#fout=open(abs_file_path+"1", 'w+')   write into this file for weka
for i in range(1,num_lines):
    cols = lines[i].split("\t")
    cols[0] = analysis.cleanup(cols[0])      #write to a file new cleaned things 
    #unigrams
    words_filtered=[]   #remove words less than 2 letters in length
    words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
    #tweets.append((words_filtered,cols[1]))
    quadgrams_list = analysis.generate_ngrams(4, cols[0])
    if(len(quadgrams_list) > 0):
        tweets.append((quadgrams_list,cols[1]))


"""
  we have words in tweets separated as ngrams and class
  TODO:  what we need use the ngrams as features:pass them through get_word_features for freq dist
  pass the frequency dist and the class as features to the classifier
  what I need/not undestand: 
  word_features has only words no class i need class for classifier modif.......
  
"""
#features for the tweets
word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
print word_features

#feature_set= [(analysis.extract_features(tweet,word_features),sentiment) for (tweet,sentiment) in tweets]
#print feature_set    
#train_set, test_set = featuresets[500:], featuresets[:500]
#classifier = nltk.NaiveBayesClassifier.train(feature_set)
tweet = 'China hands Leads'


training_set = nltk.classify.apply_features(analysis.extract_features, tweets)
print training_set
#print("Starting...")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(32)
#print("Model built...")
#tweet = 'China hands Leads'
print classifier.classify(analysis.extract_features(tweet.split())) #we have to modify this to use the ngram model as well

