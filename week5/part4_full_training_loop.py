#-----part5_full_training_loop.py-----#
#----------Andrew_Martin--------------#




"""
Assumes part2_forward_hidden.py and part3_one_backprop_step.py are already
implemented, with:
    relu(x)                                            in part2
    relu2deriv(y)                                       in part3
    one_step(layer_0, target, weights_0_1, weights_1_2, alpha)
        -> (weights_0_1, weights_1_2, layer_2, layer_2_error)  in part3
"""

import numpy as np
from part2_forward_hidden import relu
from part3_one_backprop_step import relu2deriv, one_step

# ---------------------------------------------------------
# Trial dataset (Korr's Three Pillars)
# ---------------------------------------------------------
tells = np.array([[1, 0, 1],   # foot shift, no guard drop, exhale
                   [0, 1, 1],   # no shift, guard drop, exhale
                   [0, 0, 1],   # only exhale
                   [1, 1, 1]])  # all three (the bluff)

strike = np.array([[1, 1, 0, 0]]).T  # column vector, shape (4, 1)


def train(tells, strike, alpha, epochs, hidden_size, seed):
    """
    Outer epoch loop + inner sensing loop (stochastic GD -- one update
    per sensing, via one_step from Part 3).

    Returns: (weights_0_1, weights_1_2, error_history)
    """
    np.random.seed(seed)
    n_features = tells.shape[1]

    weights_0_1 = 2 * np.random.random((n_features, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    error_history = []

    for epoch in range(epochs):
        total_error = 0.0

        for i in range(len(tells)):
            layer_0 = tells[i:i + 1]   # shape (1, 3)
            target = strike[i:i + 1]   # shape (1, 1)

            weights_0_1, weights_1_2, layer_2, layer_2_error = one_step(
                layer_0, target, weights_0_1, weights_1_2, alpha
            )

            total_error += layer_2_error

        error_history.append(total_error)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: total error = {total_error:.6f}")

    return weights_0_1, weights_1_2, error_history


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 2. Run with alpha=0.2, epochs=60, hidden_size=4, seed=1
    # ---------------------------------------------------------
    print("=== Training run (alpha=0.2, epochs=60, hidden_size=4, seed=1) ===")
    weights_0_1, weights_1_2, error_history = train(
        tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1
    )

    # ---------------------------------------------------------
    # 3. Final predictions vs goals
    # ---------------------------------------------------------
    print("\n=== Final predictions vs goals ===")
    for i in range(len(tells)):
        layer_1 = relu(tells[i:i + 1].dot(weights_0_1))
        layer_2 = layer_1.dot(weights_1_2)
        pred = layer_2.item()
        goal = strike[i].item()
        side_ok = (pred > 0.5) == (goal > 0.5)
        print(f"  Sensing {i} {tells[i]}: prediction={pred:.4f}, "
              f"goal={goal}, correct side: {side_ok}")

    # ---------------------------------------------------------
    # 4. Comment block -- read the hidden layer
    # ---------------------------------------------------------
    print("\n=== Hidden layer weights ===")
    print("weights_0_1.round(2):\n", weights_0_1.round(2))
    print("weights_1_2.round(2):\n", weights_1_2.round(2))

    # TODO 
    #
    # Hidden unit 0: ...
    # Hidden unit 1: ...
    # Hidden unit 2: ...
    # Hidden unit 3: ...
    #
  

    # ---------------------------------------------------------
    # 5. Hidden-size sweep
    # ---------------------------------------------------------
    print("\n=== Hidden-size sweep ===")
    hidden_sizes = [1, 2, 4, 8, 16]
    seeds = [1, 2, 3]
    results = {}  # (hidden_size, seed) -> final total error

    for hs in hidden_sizes:
        for s in seeds:
            _, _, errs = train(tells, strike, alpha=0.2, epochs=60,
                                hidden_size=hs, seed=s)
            results[(hs, s)] = errs[-1]

    print("\nhidden_size | seed=1     | seed=2     | seed=3")
    for hs in hidden_sizes:
        row = " | ".join(f"{results[(hs, s)]:.6f}" for s in seeds)
        print(f"{hs:>11} | {row}")

    # TODO (write this in your own words, using the table above):
    # Two sentences: which sizes FAIL on every seed, which SUCCEED on
    # every seed, and which are a COIN FLIP (succeed on some seeds, fail
    # on others)? Then explain briefly why a network might solve the
    # Trial at one seed and not another (hint: think about where random
    # initialization lands relative to ReLU's dead zone, and whether a
    # hidden layer that's too small can even represent the XOR-like
    # pattern regardless of initialization).
