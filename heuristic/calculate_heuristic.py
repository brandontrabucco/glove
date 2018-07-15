"""Author: Brandon Trabucco
Calculate the word embedding heuristic for the entire training embeddings.
"""


import glove.configuration
import glove.heuristic


config = glove.configuration.HeuristicConfiguration(
    embedding=300, 
    filedir="../embeddings/", 
    length=70000,
    start_word="<S>",
    end_word="</S>",
    unk_word="<UNK>",
    radius=20,
    distances_dir="../distances/",
    distances_threads=7,
    heuristic_dir="./")


glove.heuristic.dump(config)
