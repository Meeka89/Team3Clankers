import numpy as np

# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
                  [0, 1, 1], # no shift, guard drop, exhale
                  [0, 0, 1], # only exhale
                  [1, 1, 1]]) # all three (the bluff)

# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

def single_layer_train(tells, strike, alpha, epochs, seed):
    strike = strike.ravel()
    np.random.seed(seed)
    weights = 2 * np.random.random(3) - 1
    for epoch in range(epochs):

        total_error = 0

        for i in range(len(tells)):
            pred = np.dot(tells[i], weights)
            delta = pred - strike[i]
            weights -= alpha * delta * tells[i]
            total_error += delta ** 2

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}",
                  f"error: {total_error}")

    print(f"Final Weights: {weights}")
    for i in range(len(tells)):
        final_pred = np.dot(tells[i], weights)
        print(f"Sensing {i}: Prediction = {final_pred}, Goal = {strike[i]}")


if __name__ == "__main__":
    single_layer_train(tells, strike, 0.1, 60, 1)

# why did the training fail?
# A. The training failed because the exhale column added no useful information.
# since it was always 1. Foot shift and guard drop could also not predict the target
# by themselves because each was on for both strikes and holds.
# The actual pattern depended on both of them. A single weighted sum cannot represent this pattern.
