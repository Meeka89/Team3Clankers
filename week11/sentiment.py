"""Part 2: learn word embeddings by classifying movie-review sentiment."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_sentiment(input_dataset, labels, vocab, train_count=24000,
                    iterations=2):
    """Train on the first train_count reviews; evaluate on the remainder.

    Return the word embeddings, output weights, and held-out accuracy.
    Smaller splits can be supplied by the integration tests.
    """
    if len(input_dataset) != len(labels):
        raise ValueError("Each review must have one label.")
    if not 0 < train_count < len(input_dataset):
        raise ValueError("The split must include training and test reviews.")

    np.random.seed(1)
    alpha, hidden_size = 0.04, 100
    weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

    # A word contributes once per review; sorted indices are reproducible.
    reviews = [sorted(set(review)) for review in input_dataset]

    for epoch in range(iterations):
        correct = 0
        for i in range(train_count):
            x, y = reviews[i], labels[i]

            # Look up the review's word vectors and sum into one hidden vector.
            layer_1 = sigmoid(weights_0_1[x].sum(axis=0))
            layer_2 = sigmoid(layer_1.dot(weights_1_2))
            correct += int((layer_2.item() >= 0.5) == y)

            # Compute both deltas before changing either weight matrix.
            layer_2_delta = layer_2 - y
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)
            layer_1_delta *= layer_1 * (1 - layer_1)

            # Only word rows present in this review receive an update.
            weights_0_1[x] -= alpha * layer_1_delta
            weights_1_2 -= alpha * np.outer(layer_1, layer_2_delta)

        print(f"Epoch {epoch + 1}: training accuracy = "
              f"{correct / train_count:.3%}")

    correct = 0
    for i in range(train_count, len(reviews)):
        layer_1 = sigmoid(weights_0_1[reviews[i]].sum(axis=0))
        layer_2 = sigmoid(layer_1.dot(weights_1_2))
        correct += int((layer_2.item() >= 0.5) == labels[i])

    test_accuracy = correct / (len(reviews) - train_count)
    print(f"Test accuracy = {test_accuracy:.3%}")
    return weights_0_1, weights_1_2, test_accuracy


def main():
    from preprocess import load

    vocab, word2index, as_sets, ordered, targets = load()
    print(f"Vocabulary: {len(vocab)}; reviews: {len(targets)}")
    started = perf_counter()
    weights_0_1, weights_1_2, accuracy = train_sentiment(
        as_sets, targets, vocab
    )
    print(f"Training and evaluation time: {perf_counter() - started:.1f}s")

    # Keep generated artifacts beside this script regardless of working directory.
    output_dir = Path(__file__).resolve().parent
    np.save(output_dir / "sentiment_embeddings.npy", weights_0_1)
    with (output_dir / "sentiment_vocab.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({"vocab": vocab, "word2index": word2index}, file)
    print("Saved sentiment_embeddings.npy and sentiment_vocab.json")


if __name__ == "__main__":
    main()
