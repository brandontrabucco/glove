"""Author: Brandon Trabucco
Calculate the closest neighbors of each word in embedding space.
"""


import numpy as np
import os.path
import collections
import nltk
import threading


np.random.seed(1234567)


import glove


def _compute_neighbors(thread_index, ranges, embeddings, 
                       neighbor_ids, neighbor_distances):
    """Computes the k closest words to a set fo words in embedding space.
    Args:
        thread_index: int
        ranges: Bounds of word ids to compute
        embeddings: The emebedding matrix
        neighbors_ids: The word ids of the k closest neighbors.
        neighbors_distances: The distances corresponding to neighbors_ids
    """

    length = ranges[thread_index][1] - ranges[thread_index][0]
    print("Starting thread %d with %d words." % (thread_index, length))
    for idx in range(*ranges[thread_index]):
        embedded = embeddings[idx:(idx + 1), :]
        distances = np.linalg.norm(embeddings - embedded, axis=1)
        closest = np.argsort(distances)[:neighbor_ids.shape[1]]
        distances = distances[closest]
        neighbor_ids[idx, :] = closest
        neighbor_distances[idx, :] = distances
        if (idx - ranges[thread_index][0]) % max(1, length // 10) == 0:
            print("Thread %d processed %d words of %d." % (
                thread_index, idx - ranges[thread_index][0], length))
    print("Thread %d has finished." % thread_index)


def dump(config):
    """Loads word embeddngs an calculates neighbors.
    Args:
        config: an instance of NeighborConfiguration
    """

    vocab, embeddings = glove.load(config)
    num_threads = min(config.length, config.distances_threads)
    spacing = np.linspace(0, config.length, num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    neighbor_ids = np.zeros([config.length, config.radius]).astype(np.int)
    neighbor_distances = np.zeros([config.length, config.radius]) 

    print("Launching %d threads for calculating neighbors." % (num_threads + 1))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, embeddings, neighbor_ids, neighbor_distances)
        t = threading.Thread(target=_compute_neighbors, args=args)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    ids_name = os.path.join(config.distances_dir, "neighbor.%s.%s.%s.ids.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))
    distances_name = os.path.join(config.distances_dir, "neighbor.%s.%s.%s.distances.txt" % (
        str(config.embedding) + "d", str(config.length) + "w", str(config.radius) + "k"))
    np.savetxt(ids_name, neighbor_ids)
    np.savetxt(distances_name, neighbor_distances)
    print("Finished saving %s and %s." % (ids_name, distances_name))


def dump_default():
    """Loads the default word embeddings to compute neighbors of words.
    """

    config = glove.configuration.NeighborConfiguration(
        embedding=50, 
        filedir="./embeddings/", 
        length=12000,
        start_word="<S>",
        end_word="</S>",
        unk_word="<UNK>",
        radius=20,
        distances_dir="./distances/",
        distances_threads=7)

    return dump(config)

