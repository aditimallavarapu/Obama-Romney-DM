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
import xlrd


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
        wordlist = nltk.FreqDist(wordlist)
        hapaxes = wordlist.hapaxes()           
        features_final= [word for word in wordlist if word not in hapaxes]
        #print features_final
        
        return features_final
        
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
        
    def xls_to_txt(self, filename):
        x =  xlrd.open_workbook(filename, encoding_override = "utf-8")
        x1 = x.sheet_by_index(0)
        x2 = x.sheet_by_index(1)
        
        obama_file = open('Obama_data.txt', 'wb')
        for rownum in xrange(2,x1.nrows):
            obama_file.write(u'\t'.join([i if isinstance(i, basestring) else str(int(i)) for i in x1.row_values(rownum, 3, 5)]).encode('utf-8').strip()+ "\t\n")
        obama_file.close()
        """
        romney_file = open('Romney_data.txt', 'wb')
        for rownum in xrange(2,x2.nrows):
            romney_file.write(u'\t'.join([i if isinstance(i, basestring) else str(int(i)) for i in x2.row_values(rownum, 3, 5)]).encode('utf-8').strip()+ "\t\n")
        romney_file.close()
        """
        
    def read_file(self, filename):
        rel_path = filename
        abs_file_path = os.path.join(script_dir, rel_path)
        f = open ( abs_file_path )
        tweets=[]
        for line in f.readlines():
            cols = line.split("\t")
            cols[0] = analysis.cleanup(cols[0])      #write to a file new cleaned things 
            #unigrams
            words_filtered=[]   #remove words less than 2 letters in length
            words_filtered =[e.lower() for e in cols[0].split() if len(e)>2]      #initialise the frequency counts
            tweets.append((words_filtered,cols[1]))
    
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
        f.close()
        return tweets
   
analysis = TwitterSentimentAnalysis()
script_dir = os.path.dirname("") #<-- absolute dir the script is in
"""Read actual file have file name here"""
analysis.xls_to_txt('training-Obama-Romney-tweets.xlsx')
print("Text file saved")
#rel_path = "Obama_data.txt"
tweets = analysis.read_file("Obama_data.txt")

word_features = analysis.get_word_features(analysis.get_words_in_tweets(tweets))
print len(word_features)

training_set = nltk.classify.apply_features(analysis.extract_features, tweets)
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
