import math
import numpy as np
from collections import Counter

def analogy(positive, negative, word2index, weights_0_1, k=5):
    """
    Computes vector arithmetic (positive - negative) and returns the top k 
    nearest words excluding the input words.
    """
    # Create target vector q initialized to zeros
    q = np.zeros(weights_0_1.shape[1])
    
    # Add positive terms
    for w in positive:
        q += weights_0_1[word2index[w]]
        
    # Subtract negative terms
    for w in negative:
        q -= weights_0_1[word2index[w]]
        
    scores = Counter()
    for word, idx in word2index.items():
        # Compute Euclidean distance to vector q
        d = weights_0_1[idx] - q
        scores[word] = math.sqrt(np.sum(d * d))
        
    # Exclude input words from the result
    return [
        w for w, _ in scores.most_common() 
        if w not in positive and w not in negative
    ][:k]