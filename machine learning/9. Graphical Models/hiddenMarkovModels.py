# Code for part-of-speech tagging using HMMs

from nltk.tag import hmm
from nltk.corpus import treebank

# Prepare training data
train_data = treebank.tagged_sents()[:3000]

# Train the HMM
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

# Tag a new sentence
sentence = 'The quick brown fox jumps over the lazy dog'.split()
tagged_sentence = tagger.tag(sentence)

print("Tagged Sentence:", tagged_sentence)
