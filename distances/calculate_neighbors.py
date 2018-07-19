"""Author: Brandon Trabucco
Calculate the word embedding neighbors for the entire training embeddings.
"""


import glove.configuration
import glove.neighbors


config = glove.configuration.NeighborConfiguration(
    embedding=300, 
    filedir="../embeddings/", 
    length=70000,
    start_word="<S>",
    end_word="</S>",
    unk_word="<UNK>",
    radius=20,
    distances_dir="./",
    distances_threads=7)


glove.neighbors.dump(config)
