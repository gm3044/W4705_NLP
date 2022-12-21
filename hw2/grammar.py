#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
import collections
from math import fsum
import math

def unique(wlist) :
    unique = [] 
    for i in wlist:
        if not i in unique:
            unique.append(i)
    return unique  

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)   
        
        
        
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)
    

    
    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        LHS_pass=[]   
        LHS = list(self.lhs_to_rules.keys())
        RHS = list(self.rhs_to_rules.keys())
        
        # check whether or not LHS/RHS grammar are CNF form
        
        for i in range(len(RHS)):
            if len(RHS[i])==2:
                if str(RHS[i][0]) == str(RHS[i][0]).lower()                or str(RHS[i][1]) == str(RHS[i][1]).lower():
                    print("Error found")
            elif len(RHS[i])==1 and str(RHS[i][0])!= '0' and str(RHS[i][0])!= '.':
                if str(RHS[i][0]) == str(RHS[i][0]).upper():
                    print("Error found")
            elif len(RHS[i])>2 or len(RHS[i])==0:
                    print("Error found")

        for i in range(len(LHS)):
            if str(LHS[i]) != str(LHS[i]).upper():
                print("Error found")

             # store all the lhs to LHS_pass if their probabilties sums to one (approximately)
        for i in range(len(self.lhs_to_rules.keys())):
            sum_prob = 0 
            for j in range(len(self.lhs_to_rules[LHS[i]])):
                sum_prob+=self.lhs_to_rules[LHS[i]][j][2]
            if math.isclose(sum_prob, 1, abs_tol = 0.01): 
                LHS_pass.append(LHS[i])
                
         # if there is any elements in lhs but not in LHS_pass, then error detected
        if (collections.Counter(self.lhs_to_rules.keys()) != collections.Counter(LHS_pass))==True:
            print("Error found") 
        # if both conditions have checked and no error is found return True
        return True
        

        

if __name__ == "__main__":
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        print(grammar.verify_grammar())
    
    

