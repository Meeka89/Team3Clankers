"""Unit tests for Week 12 — Visions.

Run with: python test_visions.py

The tests use the same small, deterministic style as test_watchers.py.
The bAbI data file must be at data/qa1_train.txt beside preprocess.py.
"""

import sys

import numpy as np

from part2_The_Forward_Pass import init_weights, predict, softmax
from rnn_bptt import backward, train_step


def test_preprocessing():
    from preprocess import tokens, vocab, word2index, words2indices

    assert len(tokens) == 1010
    assert len(vocab) == 82
    assert vocab == sorted(vocab)

    # A few known words should round-trip through the vocabulary mapping.
    words = ["mary", "moved", "to", "the", "bathroom."]
    indices = words2indices(words)
    assert [vocab[index] for index in indices] == words
    assert all(word2index[word] == index for word, index in zip(words, indices))

    print("PASS: preprocessing")


def test_softmax():
    # A large input checks that softmax subtracts the maximum before exp().
    probabilities = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-9)
    assert np.all(np.isfinite(probabilities))

    uniform = softmax(np.zeros(82))
    assert np.allclose(uniform, np.full(82, 1 / 82), atol=1e-9)

    print("PASS: softmax")


def test_forward_shape():
    from preprocess import tokens, words2indices

    embed, recurrent, start, decoder = init_weights(vocab_size=82, embed_size=10, seed=1)
    sent = words2indices(tokens[0][1:])

    layers, loss = predict(sent, embed, recurrent, start, decoder)
    perplexity = np.exp(loss / len(sent))

    assert len(layers) == len(sent) + 1
    assert "hidden" in layers[0]
    assert all("hidden" in layer and "pred" in layer for layer in layers[1:])
    assert 80 <= perplexity <= 84

    print(f"PASS: forward shape (perplexity {perplexity:.2f})")


def test_training_decreases_loss():
    from preprocess import tokens, words2indices

    embed, recurrent, start, decoder = init_weights(vocab_size=82, embed_size=10, seed=1)
    sent = words2indices(tokens[0][1:])

    _, initial_loss = predict(sent, embed, recurrent, start, decoder)
    initial_perplexity = np.exp(initial_loss / len(sent))

    # Train repeatedly on one fixed sentence, as required by the handout.
    for _ in range(100):
        train_step(sent, embed, recurrent, start, decoder, alpha=0.001)

    _, final_loss = predict(sent, embed, recurrent, start, decoder)
    final_perplexity = np.exp(final_loss / len(sent))

    assert final_perplexity < initial_perplexity, (
        f"Perplexity did not decrease: {initial_perplexity:.4f} -> "
        f"{final_perplexity:.4f}"
    )

    print(
        "PASS: training decreases loss "
        f"({initial_perplexity:.2f} -> {final_perplexity:.2f})"
    )


def test_hidden_delta_sum():
    from preprocess import tokens, words2indices

    embed, recurrent, start, decoder = init_weights(vocab_size=82, embed_size=10, seed=1)
    sent = words2indices(tokens[0][1:])
    assert len(sent) >= 3

    layers, _ = predict(sent, embed, recurrent, start, decoder)
    backward(layers, sent, recurrent, decoder)

    # Layer 2 is an intermediate layer for this sentence. Its hidden delta
    # must include both its own prediction error and the next-step contribution.
    layer_idx = 2
    own_prediction_piece = layers[layer_idx]["output_delta"].dot(decoder.T)
    complete_delta = layers[layer_idx]["hidden_delta"]

    assert not np.allclose(complete_delta, own_prediction_piece), (
        "Intermediate hidden_delta appears to omit the contribution "
        "from the next recurrent step."
    )

    print("PASS: hidden delta sum")


if __name__ == "__main__":
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Softmax", test_softmax),
        ("Forward shape", test_forward_shape),
        ("Training decreases loss", test_training_decreases_loss),
        ("Hidden delta sum", test_hidden_delta_sum),
    ]

    print("=" * 60)
    print("RUNNING WEEK 12 VISIONS TEST SUITE")
    print("=" * 60)

    passed_count = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("-" * 60)
    print(f"Results: {passed_count}/{len(tests)} tests passed.")
    print("=" * 60)

    if passed_count < len(tests):
        sys.exit(1)
