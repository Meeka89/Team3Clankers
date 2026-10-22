#----------forms.py---------#
# NO GRAPH SECTION AT END


import numpy as np
import matplotlib.pyplot as plt

#--sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(out):
    # out = sigmoid(x); d/dx sigmoid = s(1 - s), so the forward output already holds 
    # everything we need, with no extra exp
    return out * (1 - out)

#--tanh
def tanh(x):
    return np.tanh(x)

def tanh_deriv(out):
    # out = tanh(x); d/dx tanh = 1 - tanh^2, so it's computed straight from the stored forward output
    return 1 - out**2

#--Relu
def relu(x):
    return (x > 0) * x

def relu_deriv(out):
    # out = relu(x); out > 0 exactly when x > 0, so the output alone tells us whether the slope is 1 or 0
    return (out > 0)

#--softmax
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)   # stability: largest score becomes 0, so exp never overflows
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)  # normalize each row (each image) to sum to 1
