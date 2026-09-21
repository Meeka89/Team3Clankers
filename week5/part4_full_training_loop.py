#-----part5_full_training_loop.py-----#
#----------Andrew_Martin--------------#





import numpy as np
from part2_forward_hidden import relu
from part3_one_backprop_step import relu2deriv, one_step

# Trial dataset (Korr's Three Pillars)
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

            total_error += float(np.sum(layer_2_error ** 2))

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

   
    # 3. Final predictions vs goals
    
    print("\n=== Final predictions vs goals ===")
    for i in range(len(tells)):
        layer_1 = relu(tells[i:i + 1].dot(weights_0_1))
        layer_2 = layer_1.dot(weights_1_2)
        pred = layer_2.item()
        goal = strike[i].item()
        side_ok = (pred > 0.5) == (goal > 0.5)
        print(f"  Sensing {i} {tells[i]}: prediction={pred:.4f}, "
              f"goal={goal}, correct side: {side_ok}")

    
    # 4. Comment block -- read the hidden layer
   
    print("\n=== Hidden layer weights ===")
    print("weights_0_1.round(2):\n", weights_0_1.round(2))
    print("weights_1_2.round(2):\n", weights_1_2.round(2))

    # TODO 

# === Hidden layer weights ===
#  weights_0_1.round(2):
 # [[-0.17  0.91 -1.   -0.9 ]
 # [-0.71 -0.93 -0.63  0.9 ]
 # [-0.21 -0.03 -0.16  0.  ]]
#  weights_1_2.round(2):
 # [[-0.59]
 # [ 1.14]
 # [-0.95]
 # [ 1.11]]

# Hidden unit 0: all three incoming weights negative (-0.17, -0.71, -0.21),
#     weak/mixed outgoing weight (-0.59). no clean signal, reads as noise.
# Hidden unit 1: strong positive on foot (0.91), strong negative on guard
#     (-0.93), near-zero on exhale (-0.03), large positive out (1.14) ->
#     foot but not guard.
# Hidden unit 2: negative on all three, largest on foot (-1.00), negative
#     out (-0.95) -> fires when foot/guard are both absent, i.e. hold.
# Hidden unit 3: strong negative on foot (-0.90), strong positive on guard
#     (0.90), exhale weight is exactly 0.00 (ignored), positive out (1.11)
#     -> guard but not foot.
    
  

   
    # 5. Hidden-size sweep
 
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

     # hidden_size | seed=1     | seed=2     | seed=3
    #           1 | 2.000000   | 2.000095   | 2.000004
    #           2 | 2.000000   | 2.000000   | 0.066403
    #           4 | 0.000015   | 1.000000   | 1.000000
    #           8 | 0.000000   | 0.033719   | 0.000000
    #          16 | 0.000000   | 0.000000   | 0.000000
    #
    # hidden_size=1 fails on every seed its stuck at total error ~2.0, meaning
    # it never learns anything. with only one hidden unit there's no way
    # to represent the two-way foot XOR guard split this dataset needs,
    # while hidden_size=8 and hidden_size=16 succeed on every seed, reliably
    # driving error to ~0. hidden_size=4 is the coin flip: it converges
    # cleanly on seed=1 but gets stuck near exactly 1.0 on seeds 2 and 3, a
    # local minimum where only some sensings get classified correctly
    # so smallest that works on seed 1 is hidden_size=4, while smallest
    # that works reliably" is hidden_size=8, and the gap between them comes
    # down to random initialization, with few units, there's a real chance
    # the starting weights land in a region where one or more ReLU units
    # end up permanently dead or fail to specialize into complementary
    # detectors, trapping the network in a local minimum it can't escape
