"""Author: Brandon Trabucco.
Gets the best first scores of words and sorts then in that order.
"""


import glove
import glove.tagger
import glove.heuristic


if __name__ == "__main__":


    vocab, embeddings = glove.load_default()
    tagger = glove.tagger.load_default()
    source_words = ["a", "black", "and", "white", "spotted", "cat", 
                    "sleeping", "on", "a", "sofa", "cushion", "."]
    scores = glove.heuristic.get_best_first_scores(
        source_words, vocab, tagger)
    print(source_words)
    print(scores)
    print(list(zip(*list(sorted(list(zip(source_words, scores)), key=lambda x: -x[1]))))[0])
    