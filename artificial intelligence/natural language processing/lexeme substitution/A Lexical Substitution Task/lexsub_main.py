#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from collections import defaultdict

import numpy as np
import tensorflow

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
    candidates = set()
    lex = wn.lemmas(lemma, pos)
    for l in lex:
        s = l.synset()
        lem = s.lemmas()
        for w in lem:
            name = w.name()
            if (name != lemma):
                name =name.replace('_', ' ')
                candidates.add(name)
    return list(candidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return wn_simple_lesk_predictor(context)

def wn_frequency_predictor(context : Context) -> str:
    count = defaultdict() 
    lex = wn.lemmas(context.lemma, context.pos)
    for l in lex:
        s = l.synset()
        lem = s.lemmas()
        for w in lem:
            name = w.name()
            if (name != context.lemma):
                name.replace('_', ' ')
                if name not in count.keys():
                    count[name] = 1
                else:
                    count[name] = 1
    candidate = max(count.items(), key=lambda x:x[1])[0]
    return candidate  

def wn_simple_lesk_predictor(context : Context) -> str:
    scores = defaultdict()
    lex = wn.lemmas(context.lemma, context.pos)
    stop_words = stopwords.words('english')
    for l in lex:
        s = l.synset()
        definitions = s.definition()
        examples = s.examples()
        hypernyms = s.hypernyms()
        definition = []
        for word in stop_words:
            for d in definitions:
                d.replace(word, '')
                definition += tokenize(d)
            for eg in examples:
                eg.replace(word, '')
                definition += tokenize(eg)
            for h in hypernyms:
                hDefs = h.definition()
                for hDef in hDefs:
                    hDef.replace(word, '')
                    definition += tokenize(hDef)
                hExamples = h.examples()
                for hEg in hExamples:
                    hEg.replace(word, '')
                    definition += tokenize(hEg)            
        con = context.left_context + context.right_context
        overlap = [x for x in definition if x in con]
        a = (2*len(overlap))/(len(definition)+len(con))
        lem = s.lemmas()
        b = 0
        c = 0
        for word in lem:
            name = word.name()
            name.replace('_', ' ')
            if name == context.lemma:
                b += 1
            else:
                c += 1     
                scores[name] = c
        for k in scores.keys():
            scores[k] = 1000*a+100*b+c 
    candidate = max(scores.items(), key=lambda x:x[1])[0]
    return candidate

   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        words = get_candidates(context.lemma, context.pos)
        stop_words = stopwords.words('english')
        similarities = defaultdict()
        for w in words:
            if w in stop_words:
                continue
            w = w.replace(' ', '_')
            x = context.lemma.replace(' ', '_')
            try:
                v1 = self.model.get_vector(w)
                v2 = self.model.get_vector(x)
            except KeyError:
                continue
            similarities[w] = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        candidate = max(similarities.items(), key=lambda x:x[1])[0]   
        return candidate 


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        input = " ".join(map(str,context.left_context))
        input += " " + context.lemma + " "
        input += " ".join(map(str,context.right_context))
        input_toks = self.tokenizer.tokenize(input)
        idx = input_toks.index(context.lemma)
        input_toks[idx] = '[MASK]'
        input_toks = self.tokenizer.encode(input_toks)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx+1])[::-1] # Sort in increasing order
        words = self.tokenizer.convert_ids_to_tokens(best_words)
        for w in words:
            if w in candidates:
                return w

class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        input = " ".join(map(str,context.left_context))
        input += " " + context.lemma + " "
        input += " ".join(map(str,context.right_context))
        input_toks = self.tokenizer.tokenize(input)
        idx = input_toks.index(context.lemma)
        input_toks[idx] = '[MASK]'
        input_toks = self.tokenizer.encode(input_toks)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx+1])[::-1] # Sort in increasing order
        words = self.tokenizer.convert_ids_to_tokens(best_words)
        for w in words:
            if w in candidates:
                return w

class customPredictor(object):
        def __init__(self):
            self.Bert = BertPredictor()
            self.W2VS = Word2VecSubst('GoogleNews-vectors-negative300.bin.gz')
        
        def predict(self, contect: Context) -> str:
            # Choose word with highest count from words predicted (by best predictors)
            p1 = wn_simple_lesk_predictor(context)
            p2 = self.Bert.predict(context)
            p3 = self.W2VS.predict_nearest(context)
            pred = [p1, p2, p3]
            return max(pred, key=pred.count)
            

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        prediction = predictor.predict(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
