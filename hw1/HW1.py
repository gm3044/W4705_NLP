#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sys,collections
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    sequence.insert(0,'START')
    sequence.append('STOP')
    if n==3: sequence.insert(0,'START')
    sequences = tuple([(sequence[i:]) for i in range(n)])
    ngrams =  list(zip(*sequences)) 
    return ngrams



class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        
        self.unigramcounts = {} 
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.tot = self.sum_1(self.unigramcounts)
        
    
    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        for word in corpus:
            temp_1 = tuple(get_ngrams(word,1))
            temp_2 = tuple(get_ngrams(word,2))
            temp_3 = tuple(get_ngrams(word,3))

            for i in temp_1:
                if i not in self.unigramcounts:
                    self.unigramcounts[i] = 1
                else:
                    self.unigramcounts[i] += 1
            for j in temp_2:
                if j not in self.bigramcounts:
                    self.bigramcounts[j] = 1
                else:
                    self.bigramcounts[j] += 1
            for k in temp_3:
                if k not in self.trigramcounts:
                    self.trigramcounts[k] = 1
                else:
                    self.trigramcounts[k] += 1

        return self.unigramcounts,self.bigramcounts,self.trigramcounts

    def sum_1(self,count):
        temp = 0
        for i in count.values():
            temp+=i
        return temp   

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram in self.trigramcounts.keys():
            trigram_prob = self.trigramcounts[trigram] / self.bigramcounts[trigram[:-1]] 
        else: 
            trigram_prob = 0

        return trigram_prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram in self.bigramcounts.keys():
            bigram_prob = self.bigramcounts[bigram] / self.unigramcounts[bigram[:-1]] 
        else: 
            bigram_prob = 0
            
        return bigram_prob


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram in self.unigramcounts.keys():
            unigram_prob = self.unigramcounts[unigram] / self.tot
        else: 
            unigram_prob = 0
            
        return unigram_prob

    # def generate_sentence(self,t=20): 
    #     """
    #     COMPLETE THIS METHOD (OPTIONAL)
    #     Generate a random sentence from the trigram model. t specifies the
    #     max length, but the sentence may be shorter if STOP is reached.
    #     """
    #     return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        smoothed_prob = lambda1*self.raw_trigram_probability(trigram) +         lambda2*self.raw_bigram_probability(trigram[1:]) +         lambda3*self.raw_unigram_probability(trigram[2:])

        return smoothed_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        temp = tuple(get_ngrams(sentence,3))
        log_prob = 0
        
        for i in temp:
                log_prob+=math.log2(self.smoothed_trigram_probability(i))
        return log_prob


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        perp = 0
        M = 0
        for sentence in corpus:
            log = self.sentence_logprob(sentence)
            perp += log
            for word in sentence:
                if word!='START': M +=1 
        pp = 2**-(perp/M)
        return pp

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    correct = 0
    total = 0
    pp_h  = []
    pp_l  = []
    for f in os.listdir(testdir1):
        pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_1<pp_2:
            correct +=1
        total+=1

    for f in os.listdir(testdir2):
        pp_3 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp_4 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        if pp_4<pp_3:
            correct +=1
        total+=1   
    return  str(round((correct/total)*100,4)) + "%"


if __name__ == "__main__":
    
# Answers for parts 1 and 2 please see my output below
# Answers for parts 3-6 please see my code above for the details


    train = "brown_train.txt"
    test = "brown_test.txt"
    model = TrigramModel(train)

    
# Testing perplexity:    

# # 1. Perplexity of Training dataset:     
    
pp_train = model.perplexity(corpus_reader(train, model.lexicon))
print(pp_train)

# 2. Perplexity of Testing dataset:    

pp_test = model.perplexity(corpus_reader(test, model.lexicon))
print(pp_test)


# Essay scoring experiment, part 7's accuracy rate:

train_h = "train_high.txt"
train_l = "train_low.txt"
testdir1 = "test_high"
testdir2 = "test_low"

acc = essay_scoring_experiment(train_h, train_l, testdir1, testdir2)
print(acc)
                               








# ### Part 1's answer

# In[2]:


get_ngrams(["natural","language","processing"],1)


# In[3]:


get_ngrams(["natural","language","processing"],2)


# In[4]:


get_ngrams(["natural","language","processing"],3)


# ### Part 2's answer

# In[58]:


model.trigramcounts[('START','START','the')]


# In[59]:


model.bigramcounts[('START','the')]


# In[60]:


model.unigramcounts[('the',)]

