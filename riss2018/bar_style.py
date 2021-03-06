"""Generates a bar chart for the synonyms statistics file.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == "__main__":
    
    with open("style.synonym_stats.json", "r") as f:
        stats = json.load(f)
        
    a = stats["mean_styled_synonyms"]
    b = stats["mean_original_synonyms"]
    c = stats["mean_dataset_synonyms"]
    
    a_std = stats["std_styled_synonyms"]
    b_std = stats["std_original_synonyms"]
    c_std = stats["std_dataset_synonyms"]
    
    fig = plt.figure()
    ax = fig.gca()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.bar([0, 1, 2], [a, b, c], tick_label=["Styled", "Original", "Ground Truth"], color =["red", "gray", "gray"])
    
    plt.ylabel("Mean Number Of Synonyms")
    plt.xlabel("Collection Of Words")

    fig.savefig("test2.png")