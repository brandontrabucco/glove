"""Author: Brandon Trabucco.
Parameters to configure the glove embeddings.
"""


import numpy as np
import os.path
import collections
import nltk


np.random.seed(1234567)


class Configuration(collections.namedtuple(
        "GloveParams", [
            "embedding", 
            "filedir", 
            "length",
            "start_word",
            "end_word",
            "unk_word"])):
    """Parameter class for glove embeddings.
    Args:
        embedding: int in [50, 100, 200, 300]
        filedir: folder containing glove.6B.<embedding>d.txt
        length: number of words in vocab.
        start_word: typycally <S>
        end_word: typically </S>
        unk_word: typically <UNK> or <?>
    """
    
    pass