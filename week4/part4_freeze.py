#-----part4_freeze.py-----#
#----------Andrew_Martin------------#

# Part 1's gradient_descent_multi lets all three weights move together.
# This part holds some of them still (frozen) and watches how the one
# weight still allowed to move behaves, compared to its unfrozen baseline
# from Part 1.

from helpers import w_sum
from part1_multi_input import gradient_descent_multi
import numpy as np


def gradient_descent_multi_frozen(input, weights, true, alpha, iterations, frozen=None):
    """
    Same contract as Part 1's gradient_descent_multi, plus `frozen`: a list
    of weight indices that are held constant (their delta is never applied).

    Returns (final_weights, error_history, weight_history), with
    weight_history[0] the starting weights, same as Part 1.
    """
    frozen = set(frozen or [])
    weights = list(weights)
    error_history = []
    weight_history = [weights.copy()]

    for _ in range(iterations):
        pred = w_sum(input, weights)
        error = (pred - true) ** 2
        delta = pred - true

        for i in range(len(weights)):
            if i not in frozen:
                weights[i] -= alpha * delta * input[i]

        weight_history.append(weights.copy())
        error_history.append(error)

    return weights, error_history, weight_history


if __name__ == "__main__":
    blade_angle, balance, breath = 8.5, 0.65, 1.2
    input_val = [blade_angle, balance, breath]
    start_weights = [0.1, 0.2, -0.1]
    true = 1.0
    alpha = 0.01
    iterations = 10

    runs = {
        "frozen=[0, 2] (only balance moves)": [0, 2],
        "frozen=[0, 1] (only breath moves)": [0, 1],
    }

    for label, frozen in runs.items():
        final_w, err_hist, w_hist = gradient_descent_multi_frozen(
            input_val, list(start_weights), true, alpha, iterations, frozen
        )
        print(f"\n{label}")
        print(f"{'iter':>4}  {'w0':>8} {'w1':>8} {'w2':>8}   {'error':>10}")
        for i, (w, e) in enumerate(zip(w_hist[1:], err_hist), start=1):
            print(f"{i:>4}  {w[0]:8.5f} {w[1]:8.5f} {w[2]:8.5f}   {e:10.6f}")
        print("Final weights:", [round(w, 5) for w in final_w])

    # unfrozen baseline, for comparison
    baseline_w, baseline_err, baseline_hist = gradient_descent_multi(
        list(input_val), list(start_weights), true, alpha, iterations
    )
    print("\nunfrozen baseline")
    print("Final weights:", [round(w, 5) for w in baseline_w])

    # --- NumPy version ---
    def gradient_descent_multi_frozen_np(input, weights, true, alpha, iterations, frozen=None):
        frozen_mask = np.ones(len(weights))
        for i in (frozen or []):
            frozen_mask[i] = 0.0

        input = np.array(input, dtype=float)
        weights = np.array(weights, dtype=float)
        error_history = []
        weight_history = [weights.copy()]

        for _ in range(iterations):
            pred = np.dot(input, weights)
            delta = pred - true
            error_history.append(delta ** 2)

            weights = weights - alpha * delta * input * frozen_mask
            weight_history.append(weights.copy())

        return weights, error_history, weight_history

    np_final_w, np_err_hist, _ = gradient_descent_multi_frozen_np(
        input_val, start_weights, true, alpha, iterations, [0, 2]
    )
    scratch_final_w, scratch_err_hist, _ = gradient_descent_multi_frozen(
        input_val, list(start_weights), true, alpha, iterations, [0, 2]
    )
    print("\nNumPy parity check (frozen=[0, 2])")
    print("weights match:", np.allclose(scratch_final_w, np_final_w))
    print("errors match:", np.allclose(scratch_err_hist, np_err_hist))

"""
5
balance, frozen=[0, 2]:  0.2 -> 0.20893  (+0.00893) over 10 iterations
balance, unfrozen:       0.2 -> 0.20123  (+0.00123) over 10 iterations
breath,  frozen=[0, 1]: -0.1 -> -0.08425 (+0.01575) over 10 iterations
breath,  unfrozen:      -0.1 -> -0.09773 (+0.00227) over 10 iterations

In both cases the free weight moves roughly 7x further when frozen than it
does in the unfrozen baseline. In the unfrozen run, blade_angle (input 8.5,
by far the largest) does most of the work of closing the gap between pred
and true, so delta shrinks quickly and every weight's update -- including
balance's or breath's -- shrinks along with it. When blade_angle is frozen,
it can't contribute to closing that gap, so delta stays larger for more
iterations, and whichever single weight is still free has to absorb all of
that lingering error by itself. Freezing a weight doesn't change how delta
is computed at any given step, but it removes that weight's contribution to
shrinking delta on the *next* step, so the remaining free weight ends up
training against a larger delta for longer.
"""
