"""Author: Brandon Trabucco
Calculate the distance heuristic for each word according to the neighbors.
"""


import numpy as np
import os.path
import collections
import nltk
import threading


np.random.seed(1234567)


import glove


def dump(config):
    """Dumps the calculated distance heuristic to a file.
    Args:
        config: an instance of HeuristicConfiguration
    """

    ids_name = os.path.join(config.distances_dir, "neighbor.%s.%s.%s.ids.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))
    distances_name = os.path.join(config.distances_dir, "neighbor.%s.%s.%s.distances.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))

    vocab, embeddings = glove.load(config)
    print("Loading %s and %s." % (ids_name, distances_name))
    neighbor_ids = np.loadtxt(ids_name, dtype=np.int32)
    neighbor_distances = np.loadtxt(distances_name, dtype=np.float32)

    heuristic_name = os.path.join(config.heuristic_dir, "heuristic.%s.%s.%s.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))
    heuristic = np.sum(neighbor_distances, axis=1)

    words_name = os.path.join(config.heuristic_dir, "heuristic.%s.%s.%s.sorted.names.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))
    words = sorted(list(range(config.length)), key=(lambda idx: heuristic[idx]))
    words = [vocab.id_to_word(idx) + "\n" for idx in words]

    np.savetxt(heuristic_name, heuristic)
    with open(words_name, "w") as f:
        f.writelines(words)
    print("Saved %s and %s." % (heuristic_name, words_name))


def dump_default():
    """Dumps the calculated distance heuristic to a file.
    """

    config = glove.configuration.HeuristicConfiguration(
        embedding=50,
        filedir="./embeddings/",
        length=12000,
        start_word="<S>",
        end_word="</S>",
        unk_word="<UNK>",
        radius=20,
        distances_dir="./distances/",
        distances_threads=7,
        heuristic_dir="./heuristic/")

    return dump(config)


def get_best_first_scores(words_list, vocab, tagger):
    """Returns the Best First Score for each word in the list.
    """

    POS_scores = {"NOUN": 11, "VERB": 10, "ADJ": 9, "NUM": 8,
        "ADV": 7, "PRON": 6, "PRT": 5, "ADP": 4,
        "DET": 3, "CONJ": 2, ".": 1, "X": 0 }
    word_tags = [POS_scores[t] / 11 for _, t in tagger.tag(words_list)]
    word_ids = [vocab.word_to_id(w) / len(vocab.reverse_vocab) for w in words_list]
    word_scores = [np.log(np.exp(x) + y) for x, y in zip(word_tags, word_ids)]
    return word_scores
