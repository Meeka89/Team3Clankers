"""
ONLY USED FOR PART 1
DO NOT MODIFY UNLESS 
AUTHORIZED TO DO SO
"""

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

    tells = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])

    for i in range(len(tells)):
        layer_0 = tells[i : i + 1]  # Keep 2D shape (1, 3)
        l1, l2 = forward(layer_0, weights_0_1, weights_1_2)

        print(f"Sensing {i}:")
        print(f"  layer_1: {l1}")
        print(f"  layer_2: {l2}\n")

"""

1. Shape Breakdown & Trace:
   - layer_0 shape: (1, 3) 
     Represents 1 input sample with 3 feature tells (foot shift, guard drop, exhale).
     
   - weights_0_1 shape: (3, 4)
     Maps the 3 input features to 4 hidden units.
     
   - layer_1 shape: (1, 4)
     Result of (1, 3) @ (3, 4) followed by elementwise ReLU activation. 
     Represents the activations of the 4 hidden units for a single sample.
     
   - weights_1_2 shape: (4, 1)
     Maps the 4 hidden unit activations to a single output prediction.
     
   - layer_2 shape: (1, 1)
     Result of (1, 4) @ (4, 1). The raw unactivated final output scalar/prediction.


2. Architecture Sketch (How Matrix Multiplications Align):

   [ Input Layer ]           [ Hidden Layer ]          [ Output Layer ]
    layer_0 (1, 3)  ----->    layer_1 (1, 4)  ----->    layer_2 (1, 1)
                          ^                         ^
                          |                         |
                   weights_0_1 (3, 4)        weights_1_2 (4, 1)

"""
