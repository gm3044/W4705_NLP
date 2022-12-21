from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import tensorflow as tf
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        self.move = {'shift':0,'left_arc':1,'right_arc':2}

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            
            inputs = self.extractor.get_input_representation(words, pos, state).reshape(1, 6)
            [predicted_outputs]= np.flip(np.argsort(self.model.predict(inputs)))

            for i in predicted_outputs[0:10]:
                # print(i)
                (action,deprel)= self.output_labels[i]
                
                if self.move[action] == 0:
                    if len(state.stack) == 0 or len(state.buffer) > 1:
                        state.shift()
#                         print(action,len(state.stack))
                        break
                    else:
                        continue
                elif self.move[action] == 1:
                    if len(state.stack)!=0 and state.stack[-1] != 0:
                        state.left_arc(deprel)
#                         print(action,state)
                        break
                    else:
                        continue

                else:
                    if len(state.stack)!=0:
                        state.right_arc(deprel)
#                         print(action,state)
                        break
                    else:
                        continue 
                        
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":
    
    tf.compat.v1.disable_eager_execution()

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
