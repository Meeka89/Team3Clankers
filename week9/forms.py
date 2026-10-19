#----------forms.py---------#
#-------Andrew_Martin-------#


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

#--The PLot

if __name__ == "__main__":
    x = np.linspace(-6, 6, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(x, sigmoid(x), label="sigmoid — range (0, 1)")
    plt.plot(x, tanh(x),    label="tanh — range (−1, 1)")
    plt.plot(x, relu(x),    label="relu — range [0, ∞)")
    for y in (-1, 0, 1):
        plt.axhline(y, linestyle="--", color="gray", linewidth=0.8)
    plt.ylim(-1.5, 3)
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.title("The Forms")
    plt.legend()
    plt.savefig("forms_plot.png", dpi=150, bbox_inches="tight")
    plt.show()