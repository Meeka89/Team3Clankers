import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward(layer_0, weights_0_1, weights_1_2):
    layer_1 = relu(layer_0 @ weights_0_1)
    layer_2 = layer_1 @ weights_1_2

    return layer_1, layer_2

if __name__ == "__main__":
    np.random.seed(1)
    hidden_size = 4
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1
