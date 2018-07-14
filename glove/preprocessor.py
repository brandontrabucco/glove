"""Author: Brandon Trabucco.
A utility class for loading glove embeddingns and vocabulary.
"""


import numpy as np
import os.path
import collections
import nltk


np.random.seed(1234567)


import glove.vocabulary
import glove.utils


class Preprocessor(object):
    """Word embedding loading class for glove."""

    def __init__(self, config):
        """Loads word embeddings and a vocabulary from a glove txt.
        Args:
             config: an instance of Configuration
        """
        
        # Assert the inputs are correct
        assert config.embedding in [50, 100, 200, 300], ""
        assert os.path.isdir(config.filedir), ""

        # Assert the targets are correct
        filename = ("glove.6B.%s.txt" % (str(config.embedding) + "d"))
        self.targetpath = os.path.join(config.filedir, filename)
        assert os.path.isfile(self.targetpath), ""
        
        self.config = config


    def run(self):
        """ Computes the vocabulary and the word embeddings matrix.
        Outputs:
            vocab: A vocabulary object usd in an image-to-text model.
            embedding_map: An initialization for the word embeddings matrix.
        """

        # Open the target file and read the embeddings
        print("Loading vocabulary from %s." % self.targetpath) 
        with open(self.targetpath, "r", encoding="UTF-8") as f:
            all_lines = f.readlines()
            all_lines = [x.strip().split(" ") for x in all_lines]
            all_names, all_vectors = zip(*[[
                x[0], np.array(x[1:]).astype(np.float32)] for x in all_lines])

        # Return a vocabulary from the embeddings
        all_names, all_vectors = list(all_names), list(all_vectors)
        all_names, all_vectors = glove.utils.check(all_names, all_vectors, self.config)
        vocab = glove.vocabulary.Vocabulary(all_names, self.config)
        embedding_map = np.stack(all_vectors)

        return vocab, embedding_map
