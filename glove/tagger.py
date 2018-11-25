"""Author: Brandon Trabucco
Calculate the part of speech tagger using the brown text corpus.
"""


import numpy as np
import os.path
import collections
import nltk
import pickle as pkl
from nltk.corpus import brown
from glove import cached_load


import glove


def dump(config):
    """Loads word embeddngs an calculates neighbors.
    Args:
        config: an instance of TaggerConfiguration
    """

    tagger_dir = config.tagger_dir
    tagger_name = os.path.join(tagger_dir, "tagger.pkl")
    os.makedirs(tagger_dir, exist_ok=True)
    if not os.path.isfile(tagger_name):
        brown_tagged_sents = brown.tagged_sents(tagset='universal')
        size = int(len(brown_tagged_sents) * 0.9)
        train_sents = brown_tagged_sents[:size]
        test_sents = brown_tagged_sents[size:]
        t0 = nltk.DefaultTagger('X')
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        t3 = nltk.TrigramTagger(train_sents, backoff=t2)
        scores = [
            [t1.evaluate(test_sents), t1], 
            [t2.evaluate(test_sents), t2], 
            [t3.evaluate(test_sents), t3]]
        best_score, best_tagger = max(scores, key=lambda x: x[0])
        print("Finished building POS tagger {0:.2f}%".format(best_score * 100))
        with open(tagger_name, 'wb') as f:
            pkl.dump(best_tagger, f)
    with open(tagger_name, 'rb') as f:
        return pkl.load(f)
    print("Finished saving %s and %s." % (ids_name, distances_name))


def dump_default():
    """Exports the trained part of speech tagger.
    """

    config = glove.configuration.TaggerConfiguration(
        tagger_dir="./tagger/")

    return dump(config)


@cached_load
def load(config):
    """Loads the trained part of speech tagger.
    """

    tagger_dir = config.tagger_dir
    tagger_name = os.path.join(tagger_dir, "tagger.pkl")
    with open(tagger_name, 'rb') as f:
        return pkl.load(f)


@cached_load
def load_default():
    """Loads the trained part of speech tagger.
    """

    config = glove.configuration.TaggerConfiguration(
        tagger_dir="./tagger/")

    return load(config)
