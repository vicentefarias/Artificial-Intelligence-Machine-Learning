import sys
from collections import defaultdict
import math
import random
import os
import os.path

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
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    ngrams = []
    ngram = ['START']
    if n == 1:
        ngrams.append(tuple(ngram))
    for word in sequence:
        if len(ngram)<n:
            while len(ngram)<n:
                ngram.append('START')
        ngram.pop(0)
        ngram.append(word)
        ngrams.append(tuple(ngram))
    stop = list(ngrams[-1])
    stop.pop(0)
    stop.append('STOP')
    ngrams.append(tuple(stop))
    return ngrams



class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 

        ##Your code here
        for line in corpus:
            unigrams = get_ngrams(line, 1)
            bigrams = get_ngrams(line, 2)
            trigrams = get_ngrams(line, 3)
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1
        self.total = sum(self.unigramcounts.values())
        self.bitotal = sum(self.bigramcounts.values())
        self.tritotal = sum(self.trigramcounts.values())
        return 
    
    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if (trigram[0], trigram[1]) == ('START', 'START'):
            p1 = self.trigramcounts[trigram]/self.unigramcounts[(trigram[0],)]
        elif self.bigramcounts[(trigram[0],trigram[1],)]==0:
            p1 = self.raw_bigram_probability((trigram[0],trigram[1],))
        else:
            p1 = self.trigramcounts[trigram]/self.bigramcounts[(trigram[0],trigram[1],)]
        return p1 

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[(bigram[0],)]!=0:
            return self.bigramcounts[bigram]/self.unigramcounts[(bigram[0],)]
        else:
            return self.raw_unigram_probability((bigram[0],))

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
        if unigram[0] in self.lexicon:
            return self.unigramcounts[unigram]/self.total
        else:
            return 1/self.total
    
    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        p1 = self.raw_trigram_probability(trigram)
        p2 = self.raw_bigram_probability((trigram[1],trigram[2],))
        p3 = self.raw_unigram_probability((trigram[2],))
        prob = lambda1*p1+lambda2*p2+lambda3*p3
        return prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        logProb = sum([math.log2(self.smoothed_trigram_probability(trigram)) for trigram in trigrams]) 
        return logProb

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = sum([self.sentence_logprob(sentence) for sentence in corpus])/self.total
        return 2**(-l)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0       

    for f in os.listdir(testdir1):
        p1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        p2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if p1 < p2:
            correct += 1
        total += 1

    for f in os.listdir(testdir2):
        p1 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        p2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        if p1 < p2:
            correct += 1
        total += 1
    
    return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)
    
    # Essay scoring experiment: 
    acc = essay_scoring_experiment('ets_toefl_data/train_high.txt', 'ets_toefl_data/train_low.txt', "ets_toefl_data/test_high", "ets_toefl_data/test_low")
    print(acc)

