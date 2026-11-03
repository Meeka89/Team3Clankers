
"""Part 3: find words with similar embeddings."""

import json
import math
from pathlib import Path

import numpy as np


def similar(target, k=10, weights_0_1=None, word2index=None, vocab=None):

    if target not in word2index:
        raise ValueError(f"Word '{target}' is not in the vocabulary.")

    target_vector = weights_0_1[word2index[target]]

    distances = {}

    for word, idx in word2index.items():
        difference = weights_0_1[idx] - target_vector
        distance = math.sqrt(np.sum(difference ** 2))
        distances[word] = distance

    nearest = sorted(distances.items(), key=lambda x: x[1])

    return [word for word, distance in nearest if word != target][:k]


def main():
    folder = Path(__file__).resolve().parent

    weights_0_1 = np.load(folder / "sentiment_embeddings.npy")

    with (folder / "sentiment_vocab.json").open(
        encoding="utf-8"
    ) as file:
        saved = json.load(file)

    word2index = saved["word2index"]

    seeds = ["terrible", "beautiful", "good"]

    for seed in seeds:
        print(f"\n{seed}:")
        print(similar(seed, 10, weights_0_1, word2index))


if __name__ == "__main__":
    main()
