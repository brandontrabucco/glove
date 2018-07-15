"""Author: Brandon Trabucco
Calculate the word embedding neighbors for the entire training embeddings.
"""


import glove.configuration
import glove.neighbors


config = glove.configuration.Configuration(
    embedding=300, 
    filedir="../embeddings/", 
    length=70000,
    start_word="<S>",
    end_word="</S>",
    unk_word="<UNK>")

glove.neighbors.dump(config)
