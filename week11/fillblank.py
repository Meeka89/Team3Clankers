#----------fillblank.py----------#
#-------Carter Owens-------------#


import numpy as np
import random
from time import perf_counter
from preprocess import load 
from sentiment import sigmoid
from similarity import similar


def build_concatenate(input_dataset):
    concatenate = np.array([idx for review in input_dataset for idx in review])
    return concatenate


def main():
    
    vocab, word2index, as_sets, ordered, targets = load()
    
    concatenated = build_concatenate(ordered)

    np.random.seed(1); random.seed(1)
    random.shuffle(ordered) # Trask shuffles the reviews once
    alpha, iterations = 0.05, 2
    hidden_size, window, neg = 50, 2, 5
    weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0 # zeros
    layer_2_target = np.zeros(neg + 1); layer_2_target[0] = 1


    started = perf_counter()
    data = ordered * iterations           
    for rev_i, review in enumerate(data):      
        for t in range(len(review)):               
            target_samples = [review[t]] + list(
                    concatenated[(np.random.rand(neg) * len(concatenated)).astype('int')])
            left = review[max(0, t-window): t]
            right = review[t+1: min(len(review), t+window)]
            layer_1 = np.mean(weights_0_1[left + right], axis=0) # (hidden,)
            layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T)) # (6,)
            layer_2_delta = layer_2 - layer_2_target
            layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])
            weights_0_1[left + right] -= layer_1_delta * alpha
            weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha 
        
        if rev_i % 1000 == 0:                                    
            print(f"\rProgress: {rev_i / len(data):6.1%}  "
                  f"({perf_counter() - started:5.0f}s)", end="")

    print(f"\nTraining time: {(perf_counter() - started) / 60:.1f} min")   

    np.save("fillblank_embeddings.npy", weights_0_1)            
    print("Saved fillblank_embeddings.npy")


    for seed in ["terrible", "beautiful", "good"]:
        neighbors = similar(seed, 11, weights_0_1, word2index)
        print(f"\n| similar('{seed}') | " + " | ".join(neighbors) + " |")


    return weights_0_1, word2index, vocab

if __name__ == "__main__":
    main()