import numpy as np
# np.array turns a list into
# an optimized NumPy vector
weights = np.array([0.1, 0.2, 0.0])
def neural_network(input, weights):
    # .dot() = dot product
    # (same as our w_sum)
    pred = input.dot(weights)
    return pred

input = np.array([8.5, 0.65, 1.2])
pred = neural_network(input,
weights)
print(pred) # 0.9800000000000001


# Pure Python -- no libraries
def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output
weights = [0.1, 0.2, 0.0]
def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred
input = [8.5, 0.65, 1.2]
pred = neural_network(input,
weights)
print(pred) # 0.9800000000000001

