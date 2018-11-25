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
    sorted_words, insertion_slots = glove.heuristic.make_insertion_sequence(
        source_words, vocab, tagger)
    print(sorted_words)
    print(insertion_slots)
    
        
        
        
        
        
        
        
        
        
        
    