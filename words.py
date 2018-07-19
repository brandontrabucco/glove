"""Author: Brandon Trabucco.
Gets the closest neighbors to the given words in embedding space.
"""


import glove
import glove.neighbors


if __name__ == "__main__":


    vocab, embeddings = glove.load_default()
    source_words = ["1", "apple", "yellow", "pizza", "cat"]
    outputs = glove.neighbors.word_neighbors(
        vocab, embeddings, source_words)


    for word, syns in zip(source_words, outputs):
        print("synonyms for %s:\n    %s" % (
            word, "\n    ".join(syns)))