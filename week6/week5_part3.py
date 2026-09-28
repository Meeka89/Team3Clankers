"""
ONLY USED FOR PART 1
DO NOT MODIFY UNLESS 
AUTHORIZED TO DO SO
"""


def relu2deriv(y):
    """
    Elementwise 1 when y > 0, 0 otherwise. Returns a NumPy array of
    the same shape as y
    """
    # y > 0 does elementwis automatically
    # change it to all ints instead of T/F
    # np array
    return np.array((y > 0).astype(int))

def relu(x):
    """Elementwise: x if x > 0, else 0."""
    return (x > 0) * x

def one_step(layer_0, target, weights_0_1, weights_1_2, alpha):
    """
    Performs one full forward → backward → update step 

    Returns (updated_weights_0_1,updated_weights_1_2, layer_2, layer_2_error)
    """
    # Forward
    layer_1 = relu(layer_0.dot(weights_0_1))
    layer_2 = layer_1.dot(weights_1_2)

    # Backward
    layer_2_error = layer_2 - target
    layer_2_delta = layer_2_error
    layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

    # Update
    updated_weights_1_2 = weights_1_2 - alpha * layer_1.T.dot(layer_2_delta)
    updated_weights_0_1 = weights_0_1 - alpha * layer_0.T.dot(layer_1_delta)

    return updated_weights_0_1,updated_weights_1_2, layer_2, layer_2_error

import numpy as np

if __name__ == "__main__":

    # 4 sensings, 3 binary tells each
    tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
                  [0, 1, 1], # no shift, guard drop, exhale
                  [0, 0, 1], # only exhale
                  [1, 1, 1]]) # all three (the bluff)
    # Ground truth: 1 = strike imminent, 0 = they will hold
    strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

    alpha = 0.2
    np.random.seed(1)
    hidden_size = 4


    layer_0 = tells[0:1]
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    layer_1_before = relu(layer_0.dot(weights_0_1))
    layer_2_before = layer_1_before.dot(weights_1_2)
    layer_2_error_before = layer_2_before - strike[0:1]

    print(f"Layer 2 Before: {layer_2_before}")
    print(f"Layer 2 Error Before: {layer_2_error_before}")
    print(f"Weights_0_1 Shape Before: {weights_0_1.shape}")
    print(f"Weights_1_2 Shape Before: {weights_1_2.shape}")

    updated_weights_0_1, updated_weights_1_2, layer_2_after, layer_2_error_after = one_step(layer_0, strike[0:1], weights_0_1, weights_1_2, alpha)

    print(f"Layer 2 After: {layer_2_after}")
    print(f"Layer 2 Error After: {layer_2_error_after}")
    print(f"Weights_0_1 Shape After: {updated_weights_0_1.shape}")
    print(f"Weights_1_2 Shape After: {updated_weights_1_2.shape}")


    # Number 4 - Pick an entry and show the calcs
    # Hand-verify: pick one entry of weights_1_2 after the step. 

    # old = weights_1_2[1][0]      = 0.75623487
    # index_1 = layer_1[0][1]     = 0.51828245
    # delta = layer_2_delta[0][0]   = -0.60805673

    # new = old - alpha * (index_1 * delta)
    #     = 0.75623487 - 0.2 * (0.51828245 * -0.60805673)
    #     = 0.75623487 - 0.2 * (-0.31515)
    #     = 0.75623487 + 0.06302
    #     = 0.81926390 = weights_1_2[1][0]


    # Number 5 - Explain the transpose
    # Explain in your own words why layer_1_delta uses weights_1_2.T (not
    # weights_1_2) and why we multiply by relu2deriv(layer_1).
   
    # Layer 1 delta has to be able to map the weights of hidden neurons to the
    # 1 output neuron, so by transposing it, we are able to match up the weight 
    # to the right contribution of the right hiddne neuron.
    # 

    # We have to multiply by the relu2derive(layer_1) because we need to account 
    # for its derivative so that we can assign the blame more correctly.