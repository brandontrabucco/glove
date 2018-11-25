"""Author: Brandon Trabucco.
Gets the closest neighbors to the given words in embedding space.
"""


import glove
import glove.configuration
import glove.neighbors


import numpy as np
import json


if __name__ == "__main__":


    config = glove.configuration.Configuration(
        embedding=300, filedir="./embeddings/", length=70000, 
        start_word="<S>", end_word="</S>", unk_word="<UNK>")
    vocab, embeddings = glove.load(config)
    
    
    source_words = ["man", "walk", "fruit", "path", "building"]
    outputs = glove.neighbors.word_neighbors(
        vocab, embeddings, source_words, k=20)
    
    
    results = [{"word": a, "synonyms": b} for a, b in zip(source_words, outputs)]


    actual_synonyms = []
    for word in results:
        actual_word = {"word": word["word"], "synonyms": []}
        
        for syn in word["synonyms"][1:]:
            answer = None
            
            while not answer:
                answer = input("Is %s a synonym for %s?  (y/n)" % (word["word"], syn))
                answer.strip().lower()
                if answer not in ["y", "n"]:
                    answer = None
                    
            if answer == "y":
                actual_word["synonyms"].append(syn)
                
        actual_synonyms.append(actual_word)
        
        
    distances = 0
    count = 0
    
    for x in actual_synonyms:
        x_embeddings = embeddings[vocab.word_to_id(x["word"]), :]
        
        for y in x["synonyms"]:
            y_embeddings = embeddings[vocab.word_to_id(y), :]
            
            distances += np.linalg.norm(x_embeddings - y_embeddings)
            count += 1
            
            
    mean_distance = distances / count
    print("Mean synonym distance: %f" % mean_distance)
    wrapper = {"threshold": mean_distance, "words": actual_synonyms}
        
        
    with open("synonyms.json", "w") as f:
        json.dump(wrapper, f)
        
    