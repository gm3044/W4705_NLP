#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

import gensim
import transformers 
import string

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    
    list_ = []
    
    model = wn.synsets(lemma, pos)
    
    for i in model:
        for j in i.lemma_names():
            if j == lemma:
                continue
            if '_' in j:
                j = j.replace('_',' ')
            if j not in list_: 
                list_.append(j)
    return list_


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    
    feq = {}
    synsets = wn.synsets(context.lemma)
#     print(sset)
    for j in synsets: 
        if synsets == []:
            continue
        for k in j.lemmas():
            if k.name() == context.lemma:
                continue
            if k.name() not in feq: 
                feq[k.name()]=k.count()
            else:
                feq[k.name()]+=k.count()
    feq_word = max(feq,key=feq.get)
   
    if '_' in feq_word:
        feq_word = feq_word.replace('_',' ')
        
    return feq_word

# +
def dic_add(synset):
    

    dic_ = set(tokenize(synset.definition()))
    dic_ = dic_.union(set(synset.lemma_names()))   
    
    for j in synset.examples():
        dic_=dic_.union(set(tokenize(j)))

    dic_=dic_.difference(stopwords.words('english'))
    
    return dic_
                  
def wn_simple_lesk_predictor(context : Context) -> str:
    
    context_words = set(context.left_context + context.right_context)
    
    max_overlap = 0
    max_count = 0
    feq_count = 0

    none_count = 0
    Non_none_count = 0
    
    result_overlap = None
    feq_synset = None
    none_result = None
    Non_none_result = None
    
    final = None
    non_final = None
    
    synsets =  wn.synsets(context.lemma)
    
    for i in synsets:
        dic_all = dic_add(i)
        hypernyms=i.hypernyms()
        if hypernyms != []:
            for j in hypernyms:
                hypernym = dic_add(j)  
                dic_all=dic_all.union(hypernym)
        overlap = len(dic_all.intersection(context_words))  
            
        if overlap > max_overlap:
                  max_overlap = overlap
                  result_overlap = i
                    
        for k in i.lemmas():
            feq_count +=k.count()
            if feq_count>=max_count and len(i.lemmas())> 1:
                max_count = feq_count
                feq_synset = i
                
    if result_overlap:
        
        for l in result_overlap.lemmas():
            if l.count() >= Non_none_count and l.name() != context.lemma:
                Non_none_count = l.count()
                Non_none_result = l.name()
                final = Non_none_result
                
    if not result_overlap and feq_synset:
    
        for n in feq_synset.lemmas():
            if n.count() >= none_count and n.name() != context.lemma:
                none_count = n.count()
                none_result =n.name() 
                Non_final = none_result
                
    try:              
        if '_' in final:
            return final.replace('_',' ')
        
        if final != None:
            return final
    
    except:
 
        return none_result


# -

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context : Context) -> str:
        
        synonyms  = get_candidates(context.lemma, context.pos)
        
        dic_ = {}
        
        for i in synonyms:
            try:
                dic_[i]=self.model.similarity(context.lemma,i)
            
            except: pass
               
        nearnest = max(dic_,key=dic_.get)
 
        return nearnest 


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model_2 = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.model_1 = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)    

    #Part 5
        
    def predict(self, context : Context) -> str:

        synonyms  = get_candidates(context.lemma, context.pos)

        sentence = "{left} {word} {right}".format(left = " ".join(context.left_context), word=context.word_form, right=" ".join(context.right_context))

        sent_mask = sentence.replace(context.word_form, "[MASK]")

       
        text = self.tokenizer.tokenize("".join(sent_mask))
        
        x = "[MASK]"
        
        if x in text:
            ind= text.index(x)
            
    
        
        input_toks = self.tokenizer.encode("".join(sent_mask))   
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model_2.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][ind+1])[::-1] # Sort in increasing order
        candidates = self.tokenizer.convert_ids_to_tokens(best_words[:10])
    
        for i in candidates:
            if i in synonyms and i != context.word_form:
                return i 
            else:
                # Note: using Word2VecSubst as a complement, BRET performs better then Word2VecSubst(0.115 vs 0.118)
                return model_1.predict_nearest(context)
            
            
            
    #Part 6  
    
    # Main reference: BERT-based Lexical Substitution, https://aclanthology.org/P19-1328.pdf
    # Key idea: Combining Cos_similarity from Word2VecSubst to BRET to find the best substitution word
    
    def part6_predict(self, context : Context) -> str:
        
        synonyms  = get_candidates(context.lemma, context.pos)


        dic_ = {}
    
        key_word=context.word_form
                
        sentence = "{left} {word} {right}".format(left = " ".join(context.left_context), word=context.word_form, right=" ".join(context.right_context))

        sent_mask = sentence.replace(context.word_form, "[MASK]")

       
        text = self.tokenizer.tokenize("".join(sent_mask))
        
        x = "[MASK]"
        
        if x in text:
            ind= text.index(x)
            
    
        input_toks = self.tokenizer.encode("".join(sent_mask))   
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model_2.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][ind+1])[::-1] # Sort in increasing order
        best_score = best_words[:10]
        candidates_2 = self.tokenizer.convert_ids_to_tokens(best_words[:30])
        candidates_1 = self.tokenizer.convert_ids_to_tokens(best_words[:10])

        for i in candidates_2:
            if i != context.word_form:
                try:
                    dic_[i]=self.model_1.similarity(context.lemma,i)

                except: pass
                
        most_likely = max(dic_,key=dic_.get)
        
        dic_final =  {}
        dic_final[most_likely]=dic_[most_likely]

        for i in candidates_1:
            if i in synonyms and i != context.lemma:
                    dic_final[i]=self.model_1.similarity(context.lemma,i)
                    final_return = max(dic_final,key=dic_final.get)
                    return  final_return       
            else:
                return model_1.predict_nearest(context)
               


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    model_1 = Word2VecSubst(W2VMODEL_FILENAME)
    model_2 = BertPredictor()
    
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = model_2.part6_predict(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
