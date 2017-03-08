# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:14:39 2017

@authors: Aditi and Suganya
"""

import os
import re
import string
import nltk
import pickle
    

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
        #cleantext = re.sub(".*\d+.*", " ", cleantext)     #replace remove numbers
        return cleantext
             
    def get_words_in_tweets(self, text):
        all_words = []
        for (words, sentiment) in text:
          all_words.extend(words)
        return all_words
    
    def get_word_features(self, wordlist):
        features={}
        wordlist = nltk.FreqDist(wordlist)
        hapa = wordlist.hapaxes()            # remove some time later
        features = wordlist.keys()
        features_final= [word for word in wordlist if word not in hapa]
        print features_final
        
        return features_final
        
    
        
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
        
    def read_file(self,data):
        filter_data=[]    
        num_lines=len(data)
        for i in range(1,num_lines-1):
            cols = data[i].split("\t")
            cols[0] = analysis.cleanup(cols[0])      #write to a file new cleaned things 
            #unigrams
            words_filtered=[]   #remove words less than 2 letters in length
            words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
            filter_data.append((words_filtered,cols[1]))
            words_filtered=[]
    
            #bigram
#            bigrams_list = analysis.generate_ngrams(2, cols[0])
#            if(len(bigrams_list) > 0):
#                tweets.append((bigrams_list,cols[1]))
#    
#           #trigram
#           trigrams_list = analysis.generate_ngrams(3, cols[0])
#           if(len(trigrams_list) > 0):
#               tweets.append((trigrams_list,cols[1]))
#           #quadgram
#           quadgrams_list = analysis.generate_ngrams(4, cols[0])
#           if(len(quadgrams_list) > 0):
#             tweets.append((quadgrams_list,cols[1]))
        return filter_data
   
analysis = TwitterSentimentAnalysis()
script_dir = os.path.dirname("") #<-- absolute dir the script is in
"""Read actual file have file name here"""
rel_path = "training_obama_tweets_nodate.txt"
abs_file_path = os.path.join(script_dir, rel_path)
f = open ( rel_path )
tweets=[]
lines = f.read().split("\n")
num_lines=len(lines)
for i in range(1,num_lines-1):
    cols = lines[i].split("\t")
    cols[0] = analysis.cleanup(cols[0])      #write to a file new cleaned things 
    #unigrams
#    words_filtered=[]   #remove words less than 2 letters in length
#    words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
#    #tweets.append((words_filtered,cols[1]))
    
    
    #bigrams
    
    bigrams_list = analysis.generate_ngrams(2, cols[0])
    if(len(bigrams_list) > 0):
        tweets.append((bigrams_list,cols[1]))


#tweets.extend(analysis.read_file(lines))

f.close()


word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
print len(word_features)

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        #features['contains(%s)' % word] = (word in document_words)
        features[word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets)
third = int(float(len(training_set)) / 3.0)
print third
train_set = training_set[0:(2*third)]
test_set = training_set[(2*third+1):]
training_set=[]
#print training_set
print("Starting...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

f = open('my_uni_classifier.pickle', 'wb')
pickle.dump(classifier,f)
f.close()

print nltk.classify.accuracy(classifier, train_set)
print nltk.classify.accuracy(classifier, test_set)
#print classifier.show_most_informative_features(32)
print("Model built...")
#tweet = '4 ppl being killed in a terrorist attack in Libya, Obama  is busy fundraising.'
#print classifier.classify(extract_features(analysis.generate_input_tokens(1, analysis.cleanup(tweet)))) 
