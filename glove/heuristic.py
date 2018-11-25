"""Author: Brandon Trabucco
Calculate the distance heuristic for each word according to the neighbors.
"""


import numpy as np
import os.path
import collections
import nltk
import threading


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


def make_insertion_sequence(words_list, vocab, tagger):
    """Generates a sequence of indices to insert the words in best first order.
    """

    scores = get_best_first_scores(words_list, vocab, tagger)
    enumerated_words = list(enumerate(words_list))
    scored_words = list(zip(enumerated_words, scores))
    sorted_scored_words = list(sorted(scored_words, key=lambda x: -x[1]))
    sorted_enumerated_words = list(list(zip(*sorted_scored_words))[0])
    sorted_words = list(list(zip(*sorted_enumerated_words))[1])

    running_caption = [sorted_enumerated_words.pop(0)]
    insertion_slots = [0]
    while len(sorted_enumerated_words) > 0:

        next_word = sorted_enumerated_words.pop(0)
        bottom_pointer = 0
        top_pointer = len(running_caption)

        while not bottom_pointer == top_pointer:

            reference_word = running_caption[(top_pointer + bottom_pointer) // 2]
            if reference_word[0] > next_word[0]:
                top_pointer = (top_pointer + bottom_pointer) // 2
            elif reference_word[0] < next_word[0]:
                bottom_pointer = (top_pointer + bottom_pointer) // 2 + 1
            else:
                top_pointer = (top_pointer + bottom_pointer) // 2
                bottom_pointer = (top_pointer + bottom_pointer) // 2

        running_caption.insert(top_pointer, next_word)
        insertion_slots.append(top_pointer)

    return sorted_words, insertion_slots
