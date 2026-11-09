"""
1. Load the trained weights from Part 4.
2. Pick three sentences from tokens[1000:1010] (past the training cut) that are declarative “X
moved/went to the Y” lines (the question lines have tab-separated answers mashed in — skip
those for this demo).
3. For each, slice [1:] to match the training regime, run predict, and print a table with three
columns: context (words seen so far), true next word, network’s prediction. Skip layer 1 in the
display (it’s the “predict from start” row and doesn’t depend on the sentence’s content).
4. Save the table to your notebook.
"""

from pathlib import Path
import numpy as np

from preprocess import tokens, vocab, words2indices
from part2_The_Forward_Pass import predict

def load_weights(path=Path(__file__).parent / "weights.npz"):
    w = np.load(path)
    return w["embed"], w["recurrent"], w["start"], w["decoder"]


def pick_statements(n=3):
   
    picked = []
    for i in range(1000, 1010):
        words = tokens[i]
        if any("\t" in w or "?" in w for w in words):
            continue
        picked.append(i)
        if len(picked) == n:
            break
    return picked


def next_word_rows(line_idx, weights):
    embed, recurrent, start, decoder = weights
    words = tokens[line_idx][1:]          # same [1:] slice as training
    sent = words2indices(words)
    layers, loss = predict(sent, embed, recurrent, start, decoder)

    rows = []
    for k in range(2, len(layers)):
        context = " ".join(words[:k - 1])
        truth = words[k - 1]
        guess = vocab[int(np.argmax(layers[k]["pred"]))]
        rows.append((context, truth, guess))
    return words, rows, float(np.exp(loss / len(sent)))


def print_table(line_idx, words, rows, ppl):
    print(f"\ntokens[{line_idx}][1:] = {words}   (perplexity {ppl:.2f})")
    print(f"{'context':<28}{'true next':<14}{'prediction':<14}")
    print("-" * 56)
    for context, truth, guess in rows:
        mark = "ok" if truth == guess else "WRONG"
        print(f"{context:<28}{truth:<14}{guess:<14}{mark}")


if __name__ == "__main__":
    weights = load_weights()
    for line_idx in pick_statements(3):
        words, rows, ppl = next_word_rows(line_idx, weights)
        print_table(line_idx, words, rows, ppl)