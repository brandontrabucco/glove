"""Author: Brandon Trabucco
Global declarations for the glove embedding library.
"""


import glove.configuration
import glove.preprocessor


def cached_load(load_fn):
    """Higher order function for caching.
    """
    
    results = None
    def cached_fn(*args, **kwargs):
        nonlocal results
        if results is None:
            results = load_fn(*args, **kwargs)
        return results
    return cached_fn


@cached_load
def load(config):
    """Load the glove embeddings and vocabulary.
    Outputs:
        vocab: A vocabulary object usd in an image-to-text model.
        embedding_map: An initialization for the word embeddings matrix.
    """
    
    return glove.preprocessor.Preprocessor(config).run()


def load_default():
    """Load the embedings using default parameters.
    Outputs:
        vocab: A vocabulary object usd in an image-to-text model.
        embedding_map: An initialization for the word embeddings matrix.
    """
    
    config = glove.configuration.Configuration(
        embedding=50, 
        filedir="./embeddings/", 
        length=12000,
        start_word="<S>",
        end_word="</S>",
        unk_word="<UNK>")
    
    return load(config)
