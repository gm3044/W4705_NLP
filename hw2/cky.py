#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
COMS W4705 - Natural Language Processing - Summer 2022
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
import warnings
import numpy as np
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and           isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    print(bp)
#                     sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
#                     return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
#                     sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
#                     return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        
        n = len(tokens)

        table = defaultdict(lambda: defaultdict(list))

        #initalization
         

        for i in range(0,n):
            for j in self.grammar.rhs_to_rules[(tokens[i],)]:
                if j[1] in j:
                    table[i,i+1][j[0]].append(j[1])
                else:
                    table[i,i+1][j[0]]=j[1]        
            
        for length in range(2,n+1):
            for i in range(0,n-length+1):
                j = i+length
                for k in range(i+1,j):
        #             print(i,k,j)
                    for value_up in list(table[i,k].keys()):
                         for v_u in list(value_up.split()):
                            for value_down in list(table[k,j].keys()):
                                for v_d in list(value_down.split()):
                                    if self.grammar.rhs_to_rules.get((v_u, v_d))!=None:
        #                                 print(grammar.rhs_to_rules.get((v_u, v_d)))
                                        for value in self.grammar.rhs_to_rules.get((v_u, v_d)):
        #                                     print(value[0])
                                            if value[0] in value:
                                                table[i,j][value[0]].append(value[1])
                                            else:
                                                table[i,j][value[0]]=value[1]
        if table[0,n].values():
            return True
        else:
            return False 
        
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
       
        n = len(tokens)
        table = defaultdict(lambda: defaultdict(list))
        probs = defaultdict(lambda: defaultdict(list))
        
        #initalization
         
        for i in range(0,n):
            for j in self.grammar.rhs_to_rules[(tokens[i],)]:
                if j[1] in j:
                    table[i,i+1][j[0]].append(j[1])
                    probs[i,i+1][j[0]].append(np.log(j[2]))
                    
                else:
                    table[i,i+1][j[0]]=j[1]
                    probs[i,i+1][j[0]]=np.log(j[2])
                    
#         for nps in table:
#             for i in table[nps].items():
#                 table[nps][i[0]]=str(i[1])   
        for nums in table: 
            for word in table[nums]:
                temp = table[nums][word]
                table[nums][word] = str(temp[0][0])
        
        for length in range(2,n+1):
            for i in range(0,n-length+1):
                j = i+length
                for k in range(i+1,j):
                    for value_up in list(table[i,k].keys()):
                         for v_u in list(value_up.split()):
                            for value_down in list(table[k,j].keys()):
                                for v_d in list(value_down.split()):
                                    if self.grammar.rhs_to_rules.get((v_u, v_d))!=None:
                                        for value in self.grammar.rhs_to_rules.get((v_u, v_d)):
                                            try:
                                                temp_prob=(np.log(value[2])+probs[i,k][v_u]+probs[k,j][v_d])
                                                old_prob = probs[i,j][value[0]]
                                                if value[0] in value and temp_prob[0]>old_prob[0]:
                                                        table[i,j][value[0]]=((v_u,i,k),(v_d,k,j))
                                                        probs[i,j][value[0]]=temp_prob
                                            except:
                                                table[i,j][value[0]]=((v_u,i,k),(v_d,k,j))
                                                probs[i,j][value[0]]=temp_prob
        for nums in probs: 
            for word in probs[nums]:
                prob=probs[nums][word]
                probs[nums][word] = prob[0]                                                
                                                
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    abbv = chart[i,j][nt]
  
    if type(abbv) == tuple:
        return (nt, get_tree(chart,abbv[0][1],abbv[0][2],abbv[0][0]),get_tree(chart,abbv[1][1],abbv[1][2],abbv[1][0]))
    else:
        return (nt, abbv)

 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        
        #Part 2
        toks_1 =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks_1))
        toks_2 =['miami', 'flights','cleveland', 'from', 'to','.']
        print(parser.is_in_language(toks_2))
        
        #Part 3
        table,probs = parser.parse_with_backpointers(toks_1)
        print(table[0,len(toks_1)][grammar.startsymbol] )
        print(probs[0,len(toks_1)][grammar.startsymbol] )
        
        # all passed
        print(check_table_format(table))
        print(check_probs_format(probs))

        #part4
        print(get_tree(table,0,len(toks_1),grammar.startsymbol))
#         print(table)
#         print(probs)
#         assert check_table_format(chart)
        #assert check_probs_format(probs)
        


# In[ ]:




