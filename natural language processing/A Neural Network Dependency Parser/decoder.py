from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0) 
        dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']
   

        while state.buffer: 
            pass
            # TODO: Write the body of this loop for part 4 
            # feature extractor to get a representation of the current state
            inRep = np.asarray([self.extractor.get_input_representation(words, pos, state)])
            # call model.predict(features)
            out = self.model.predict(inRep)
            # sort softmax activated vector of poss actions
            # iterate through possible actions
            actions = {}
            for i in range(len(out)):
                actions[i] = out[i]
            # sort possible actionsa according to highest probability
            sorted(actions.items(), key=lambda x: x[1], reverse=True)
            # check each possible action for available moves
            for k,v in actions.items():
                mv = (k/45, k%45)
                # if stack empty, shift
                if len(state.stack)==0:
                    state.shift()
                    break
                # if one word in buff and shifting, check stack is empty
                elif len(state.buffer)==1 and mv[0]==2:
                    # skip action if stack not empty
                    if len(state.stack)!=0:
                        continue
                # take action if empty stack
                    else:
                        state.shift()
                        break
                # if left-arc and root node skip action
                elif mv[0] == 0 and mv[1] == dep_relations.index('root'):
                    continue 
                else:
                    actIdx = mv[0]
                    depIdx = mv[1]
                    dep = dep_relations[depIdx]
                    if actIdx == 0:
                        state.left_arc(dep)
                    if actIdx == 1:
                        state.right_arc(dep)
                    else:
                        state.shift
                    break


        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

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
        
