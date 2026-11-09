
"""
1. Set alpha = 0.001, iterations = 30000, TRAIN_N = 1000.
2. Loop. At every step, sent = words2indices(tokens[it % TRAIN_N][1:]) (cycles through
the first 1,000 lines thirty times; tokens[1000..1009] stay out of training for Part 5).
3. Every 1,000 iterations, record (iter, perplexity) to a list.
4. After training, save the perplexity log to a CSV and save the four weight tensors with np.savez.
"""

import csv
import time
from rnn_bptt import train_step
from part2_The_Forward_Pass import init_weights
from preprocess import tokens, vocab, words2indices
import numpy as np

alpha = 0.001
iterations = 30000
TRAIN_N = 1000

embed, recurrent, start, decoder = init_weights(len(vocab))
log = []
t0 = time.time()

for it in range(iterations):
    sent = words2indices(tokens[it % TRAIN_N][1:])
    perplexity = train_step(sent, embed, recurrent, start, decoder, alpha)

    if it % 1000 == 0:
        log.append((it, perplexity))
        print(f"iter {it:>6}  perplexity {perplexity:.2f}")

print(f"Training time: {time.time() - t0:.1f}s")

with open("perplexity_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iter", "perplexity"])
    writer.writerows(log)

np.savez("weights.npz", embed=embed, recurrent=recurrent,
         start=start, decoder=decoder)
