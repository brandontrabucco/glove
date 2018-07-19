""" Author: Brandon Trabucco
Utility functions for loading vocabulary ad embeddings.
"""


import numpy as np
import os.path
import collections
import nltk


np.random.seed(1234567)


import glove.preprocessor


def check(vocab_names, embedding_vectors, config):
    """Checks if the list contains all necessary words.
    Args:
        vocab_names: Array containing the sorted names of
            the elements of the vocabulary in order
            corresponding to frequency.
        embedding_vectors: Array containing the vectors
            corresponding to each word in vocab_names
            by index.
        config: an instance of Configuration
    Outputs:
        vocab_names: Array containing the cropped and 
            added special words.
        embedding_vectors: Array containing the cropped
            embeddings for special words. 
    """
    
    # Verify there is data to work with.
    assert vocab_names is not None, ""
    assert embedding_vectors is not None, ""
    assert len(vocab_names) == len(embedding_vectors), ""

    # Report if vocab contained special_words.
    contained_start = config.start_word in vocab_names
    contained_end = config.end_word in vocab_names
    contained_unk = config.unk_word in vocab_names

    # Report how many special words must be added.
    words_to_add = 0
    if not contained_start:
        words_to_add += 1
    if not contained_end:
        words_to_add += 1
    if not contained_unk:
        words_to_add += 1

    # Crop the list of names.
    if config.length is not None:
        assert config.length >= 3 and config.length <= 400000, ""
        vocab_names = vocab_names[:(config.length - words_to_add)]
        embedding_vectors = embedding_vectors[:(config.length - words_to_add)]

    # Check is vocab contains special tokens
    if not contained_start:
        vocab_names.append(config.start_word)
        embedding_vectors.append(np.random.uniform(
            low=-0.1, high=0.1, size=config.embedding))
    if not contained_end:
        vocab_names.append(config.end_word)
        embedding_vectors.append(np.random.uniform(
            low=-0.1, high=0.1, size=config.embedding))
    if not contained_unk:
        vocab_names.append(config.unk_word)
        embedding_vectors.append(np.random.uniform(
            low=-0.1, high=0.1, size=config.embedding))
    
    return vocab_names, embedding_vectors
