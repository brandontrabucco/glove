"""Author: Brandon Trabucco.
Gets the closest neighbors to the given words in embedding space.
"""


import glove
import glove.configuration
import glove.neighbors


import numpy as np
import json
import argparse


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser("synonyms")
    parser.add_argument("--threshold", type=float, default=5.632423353947836)
    args = parser.parse_args()
    
    
    threshold = args.threshold


    config = glove.configuration.Configuration(
        embedding=300, filedir="./embeddings/", length=70000, 
        start_word="<S>", end_word="</S>", unk_word="<UNK>")
    vocab, embeddings = glove.load(config)
    
    
    with open("style.captions.json", "r") as f:
        captions = json.load(f)
        
        
    dataset_ids = []
    original_ids = []
    style_ids = []
    for e in captions:
        dataset_ids.extend([vocab.word_to_id(w) for w in e["ground_truth"].strip().lower().split()])
        original_ids.extend([vocab.word_to_id(w) for w in e["original"].strip().lower().split()])
        style_ids.extend([vocab.word_to_id(w) for w in e["styled"].strip().lower().split()])
    dataset_ids = set(dataset_ids)
    original_ids = set(original_ids)
    style_ids = set(style_ids)
    
    
    def get_synonyms(ids, vocab, embeddings, threshold):
        
        counts = []
        for x in ids:
            
            print("Beginning %d of %d" % (x, len(ids)))
            
            x_embeddings = embeddings[x, :]
            x_count = 0
            
            for y in range(70000):
                y_embeddings = embeddings[y, :]
                
                distance = np.linalg.norm(x_embeddings - y_embeddings)
                if distance < threshold:
                    x_count += 1
                    
            counts.append(x_count)
            
        return counts

    dataset_synonyms = get_synonyms(dataset_ids, vocab, embeddings, threshold)
    original_synonyms = get_synonyms(original_ids, vocab, embeddings, threshold)
    styled_synonyms = get_synonyms(style_ids, vocab, embeddings, threshold)
        
    mean_dataset_synonyms = np.mean(dataset_synonyms)
    mean_original_synonyms = np.mean(original_synonyms)
    mean_styled_synonyms = np.mean(styled_synonyms)
      
    std_dataset_synonyms = np.std(dataset_synonyms)
    std_original_synonyms = np.std(original_synonyms)
    std_styled_synonyms = np.std(styled_synonyms)
      
    synonyms_dump = {
        "mean_dataset_synonyms": mean_dataset_synonyms,
        "mean_original_synonyms": mean_original_synonyms,
        "mean_styled_synonyms": mean_styled_synonyms,
        "std_dataset_synonyms": std_dataset_synonyms,
        "std_original_synonyms": std_original_synonyms,
        "std_styled_synonyms": std_styled_synonyms}
        
    with open("style.synonym_stats.json", "w") as f:
        json.dump(synonyms_dump, f)
        
    