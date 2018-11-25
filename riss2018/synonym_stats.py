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
    
    
    with open("captions.json", "r") as f:
        captions = json.load(f)
        
        
    dataset_ids = []
    model_ids = []
    vocab_ids = [x for x in range(70000)]
    for e in captions:
        dataset_ids.extend([vocab.word_to_id(w) for w in e["ground_truth"].strip().lower().split()])
        model_ids.extend([vocab.word_to_id(w) for w in e["caption"].strip().lower().split()])
    dataset_ids = set(dataset_ids)
    model_ids = set(model_ids)
    
    
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
    model_synonyms = get_synonyms(model_ids, vocab, embeddings, threshold)
    vocab_synonyms = get_synonyms(vocab_ids, vocab, embeddings, threshold)
        
    mean_dataset_synonyms = np.mean(dataset_synonyms)
    mean_model_synonyms = np.mean(model_synonyms)
    mean_vocab_synonyms = np.mean(vocab_synonyms)
      
    std_dataset_synonyms = np.std(dataset_synonyms)
    std_model_synonyms = np.std(model_synonyms)
    std_vocab_synonyms = np.std(vocab_synonyms)
      
    synonyms_dump = {
        "mean_dataset_synonyms": mean_dataset_synonyms,
        "mean_model_synonyms": mean_model_synonyms,
        "mean_vocab_synonyms": mean_vocab_synonyms,
        "std_dataset_synonyms": std_dataset_synonyms,
        "std_model_synonyms": std_model_synonyms,
        "std_vocab_synonyms": std_vocab_synonyms}
        
    with open("synonym_stats.json", "w") as f:
        json.dump(synonyms_dump, f)
        
    