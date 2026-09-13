#-----part2_multi_output.py-----#
#----------Andrew_Martin------------#


import numpy as np

# ---------------------------------------------------------------
# Part 1: from-scratch gradient descent, one input -> three outputs
# ---------------------------------------------------------------
def gradient_descent_outputs(input, weights, trues, alpha, iterations):
    """
    input   : scalar
    weights : length-3 list of starting weights
    trues   : length-3 list of target outputs
    alpha   : learning rate
    iterations : number of training steps

    Returns (final_weights, error_history, weight_history)
      error_history  -> list of per-iteration mean squared error across the 3 outputs
      weight_history -> list of the 3 weights after each iteration
    """
    weights = list(weights)
    error_history = []
    weight_history = []

    for _ in range(iterations):
        # one shared input drives three independent predictions
        preds = [w * input for w in weights]

        # per-output raw error (pred - true), and MSE across the three outputs
        deltas = [p - t for p, t in zip(preds, trues)]
        mse = sum(d ** 2 for d in deltas) / len(deltas)
        error_history.append(mse)

        # each weight's update is scaled by the SAME input
        weight_deltas = [d * input for d in deltas]
        weights = [w - alpha * wd for w, wd in zip(weights, weight_deltas)]

        weight_history.append(list(weights))

    return weights, error_history, weight_history


# ---------------------------------------------------------------
# Part 2: train and print a clean iteration log
# ---------------------------------------------------------------
balance = [0.65, 0.20, 0.90]           # only balance[0] is used here
trues_multi = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]

input_val = balance[0]
start_weights = [0.3, 0.2, 0.9]
trues = trues_multi[0]
alpha = 0.1
iterations = 20

final_w, err_hist, w_hist = gradient_descent_outputs(
    input_val, start_weights, trues, alpha, iterations
)

print("From-scratch training log")
print(f"{'iter':>4}  {'w0':>8} {'w1':>8} {'w2':>8}   {'mse':>10}")
for i, (w, e) in enumerate(zip(w_hist, err_hist), start=1):
    print(f"{i:>4}  {w[0]:8.5f} {w[1]:8.5f} {w[2]:8.5f}   {e:10.6f}")

print("\nFinal weights:", [round(w, 5) for w in final_w])
print("Final MSE:", round(err_hist[-1], 6))


# ---numpy version---
def gradient_descent_outputs_np(input, weights, trues, alpha, iterations):
    w = np.array(weights, dtype=float)
    t = np.array(trues, dtype=float)
    error_history = []
    weight_history = []

    for _ in range(iterations):
        preds = w * input         # broadcast: scalar input, 3 weights
        deltas = preds - t
        mse = np.mean(deltas ** 2)
        error_history.append(mse)

        w = w - alpha * (deltas * input)
        weight_history.append(w.copy())

    return w, error_history, weight_history

np_final_w, np_err_hist, np_w_hist = gradient_descent_outputs_np(
    input_val, start_weights, trues, alpha, iterations
)

# parity check
w_match = np.allclose(final_w, np_final_w)
e_match = np.allclose(err_hist, np_err_hist)
print("\nNumPy parity check")
print("Final weights match:", w_match, "->", np_final_w)
print("Error histories match:", e_match)


# ---------------------------------------------------------------
# Part 4: first-iteration vs last-iteration delta ratios
# ---------------------------------------------------------------
def per_output_deltas(input, weights, trues):
    return [w * input - t for w, t in zip(weights, trues)]

first_deltas = per_output_deltas(input_val, start_weights, trues)
last_deltas  = per_output_deltas(input_val, final_w, trues)

print("\nDelta comparison (iteration 1 vs iteration 20)")
for i in range(3):
    ratio = last_deltas[i] / first_deltas[i]
    print(f"output {i}: first={first_deltas[i]:+.5f}  last={last_deltas[i]:+.5f}  ratio={ratio:.5f}")