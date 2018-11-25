"""Author: Brandon Trabucco.
Parameters to configure the glove embeddings.
"""


import numpy as np
import os.path
import collections
import nltk


np.random.seed(1234567)


def merge_fields(cls):
    """Combine multiple named tuples into a single tuple class.
    Args:
        cls: The class inheriting from multiple named tuples.
    """

    name = cls.__name__
    bases = cls.__bases__

    fields = []
    for c in bases:
        if not hasattr(c, '_fields'):
            continue
        fields.extend(f for f in c._fields if f not in fields)

    if len(fields) == 0:
        return cls

    combined_tuple = collections.namedtuple('%sCombinedNamedTuple' % name, fields)
    return type(name, (combined_tuple,) + bases, dict(cls.__dict__))


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


@merge_fields
class NeighborConfiguration(Configuration, collections.namedtuple(
        "NeighborParams", [
            "radius",
            "distances_dir",
            "distances_threads"])):
    """Parameter class for computing word neighbors.
    Args:
        radius: int number of closest neighbors to consider
        distances_dir: folder to dump the neighbors calculations
        distances_threads: number of helper threads to spawn (7)
    """

    pass


class TaggerConfiguration(collections.namedtuple(
        "Tagger", [
            "tagger_dir"])):
    """Parameter class for computing the part of speech tagger.
    Args:
        tagger_dir: folder to dump the part of speech tagger
    """

    pass


@merge_fields
class HeuristicConfiguration(NeighborConfiguration, collections.namedtuple(
        "HeuristicParams", [
            "heuristic_dir"])):
    """Parameter class for computing word neighbors.
    Args:
        heuristic_dir: folder to dump the heuristic calculations
    """

    pass
