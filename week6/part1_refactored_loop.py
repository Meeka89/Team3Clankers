#-----part1_refactored_loop.py--------#
#----------Andrew_Martin--------------#

import numpy as np
from week5_part2 import relu
from week5_part3 import relu2deriv, one_step


tells = np.array([[1, 0, 1],   # foot shift, no guard drop, exhale
                   [0, 1, 1],   # no shift, guard drop, exhale
                   [0, 0, 1],   # only exhale
                   [1, 1, 1]])  # all three (the bluff)

strike = np.array([[1, 1, 0, 0]]).T  # column vector, shape (4, 1)


def train(tells, strike, alpha, epochs, hidden_size, seed):
    """
    Same Week 5 algorithm (single hidden layer, ReLU on the hidden layer,
    no output activation, stochastic gradient descent, one update per
    sensing), rewritten so each pass through the network is organized into
    the four movements from lecture: FORWARD, COMPARE, BACKWARD, UPDATE.

    Returns: (weights_0_1, weights_1_2, error_history)
    """
    np.random.seed(seed)
    n_features = tells.shape[1]

    weights_0_1 = 2 * np.random.random((n_features, hidden_size)) - 1  # (n_features, hidden_size)
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1           # (hidden_size, 1)

    error_history = []

    # Tuesday's lecture loop (reference) had the same four movements:
    # FORWARD -> COMPARE -> BACKWARD -> UPDATE, just written for one
    # fixed hidden_size instead of a variable one.

    for epoch in range(epochs):
        total_error = 0.0

        for i in range(len(tells)):
            layer_0 = tells[i:i + 1]   # (1, n_features)
            target = strike[i:i + 1]   # (1, 1)

            # --- FORWARD ---
            layer_1 = relu(layer_0 @ weights_0_1)  # (1, hidden_size) or (1,3) but its left as hidden_size so
                                                   #                  it isnt hardcoded. I did similar things with     
            layer_2 = layer_1 @ weights_1_2        # (1, 1)           lines 30, 29, 42, 55, 58, and 59

            # --- COMPARE ---
            layer_2_error = (layer_2 - target) ** 2   # (1, 1)        
            layer_2_delta = layer_2 - target          # (1, 1)

            # --- BACKWARD ---
            layer_1_delta = (layer_2_delta @ weights_1_2.T) * relu2deriv(layer_1)  # (1, hidden_size)

            # --- UPDATE ---
            weights_1_2 -= alpha * (layer_1.T @ layer_2_delta)  # (hidden_size, 1)
            weights_0_1 -= alpha * (layer_0.T @ layer_1_delta)  # (n_features, hidden_size)

            total_error += float(np.sum(layer_2_error))

        error_history.append(total_error)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: total error = {total_error:.6f}")

    return weights_0_1, weights_1_2, error_history


if __name__ == "__main__":

    # 2. Run with alpha=0.2, epochs=60, hidden_size=4, seed=1

    print("=== Training run (alpha=0.2, epochs=60, hidden_size=4, seed=1) ===")
    weights_0_1, weights_1_2, error_history = train(
        tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1
    )

    print(f"\nFinal-epoch error: {error_history[-1]:.6f}")
    print("Expected (Week 5 version): 0.000015")

    # 3. Final predictions vs goals

    print("\n=== Final predictions vs goals ===")
    for i in range(len(tells)):
        layer_1 = relu(tells[i:i + 1] @ weights_0_1)
        layer_2 = layer_1 @ weights_1_2
        pred = layer_2.item()
        goal = strike[i].item()
        side_ok = (pred > 0.5) == (goal > 0.5)
        print(f"  Sensing {i} {tells[i]}: prediction={pred:.4f}, "
              f"goal={goal}, correct side: {side_ok}")

    print("\n=== Hidden layer weights ===")
    print("weights_0_1.round(2):\n", weights_0_1.round(2))
    print("weights_1_2.round(2):\n", weights_1_2.round(2))



## RESULTS! ##
#-------------------
# week5 results: 
# Epoch 1: total error = 1.414206
# Epoch 10: total error = 0.634231
# Epoch 20: total error = 0.358384
# Epoch 30: total error = 0.083018
# Epoch 40: total error = 0.006467
# Epoch 50: total error = 0.000329
# Epoch 60: total error = 0.000015 <-- what we want to match up
# week6 results:
# === Training run (alpha=0.2, epochs=60, hidden_size=4, seed=1) ===
# Epoch 1: total error = 1.414206
# Epoch 10: total error = 0.634231
# Epoch 20: total error = 0.358384
# Epoch 30: total error = 0.083018
# Epoch 40: total error = 0.006467
# Epoch 50: total error = 0.000329
# Epoch 60: total error = 0.000015 <-- here we see that the refacted loop is
#                                      pretty much identical the origanl week5 training loop