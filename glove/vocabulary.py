"""Author: Brandon Trabucco.
A utility class for manipulating a word vocabulary. 
"""


import numpy as np
import os.path
import collections
import nltk


np.random.seed(1234567)


class Vocabulary(object):
    """Vocabulary class for word embeddings."""

    def __init__(self, vocab_names, config):
        """Initializes the vocabulary.
        Args:
            vocab_names: Array containing the sorted names of
                the elements of the vocabulary in order
                corresponding to frequency.
            config: an instance of Configuration
        """
        
        assert vocab_names is not None, ""
        assert config.start_word in vocab_names, ""
        assert config.end_word in vocab_names, ""
        assert config.unk_word in vocab_names, ""

        # Reverse the vocabulary mapping
        vocab = dict([(x, y) for (y, x) in enumerate(vocab_names)])
        print("Created vocabulary with %d names." % len(vocab_names))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = vocab_names  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_id = vocab[config.start_word]
        self.end_id = vocab[config.end_word]
        self.unk_id = vocab[config.unk_word]
        
        self.config = config
        

    def word_to_id(self, word):
        """Returns the integer word id of a word string.
        """
        
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id


    def id_to_word(self, word_id):
        """Returns the word string of an integer word id.
        """
        
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]
        
        
    def cleanup_chunk(self, text_chunk):
        """Utility for cleaning up a block of text.
        Args:
            Unformatted block of text.
        Outputs:
            Tuple of nested list fo words, and nested list of ids.
        """
        
        text_chunk = text_chunk.strip().lower()
        sentences = nltk.tokenize.sent_tokenize(text_chunk)
        words = [nltk.tokenize.word_tokenize(s) for s in sentences]
        for s in words:
            s.insert(0, self.config.start_word)
            s.append(self.config.end_word)
            for i, w in enumerate(s):
                if self.word_to_id(w) is self.unk_id:
                    s[i] = self.config.unk_word
        tokens = [[self.word_to_id(w) for w in s] for s in words]
        return words, tokens
