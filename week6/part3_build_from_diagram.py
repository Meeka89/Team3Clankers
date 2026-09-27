import numpy as np

# ==============================================================================
# 4. Comment block explaining shape constraints for each weight matrix:
# ------------------------------------------------------------------------------
# weights_0_1 shape (3, 8): Connects layer_0 (1, 3) to layer_1 (1, 8). 
#   The input dimension must match layer_0's feature count (3) to perform 
#   matrix multiplication, while the output dimension sets layer_1's width (8).
#
# weights_1_2 shape (8, 4): Connects layer_1 (1, 8) to layer_2 (1, 4).
#   The input dimension must match layer_1's hidden size (8), and the output 
#   dimension sets layer_2's width (4).
#
# weights_2_3 shape (4, 1): Connects layer_2 (1, 4) to layer_3 (1, 1).
#   The input dimension must match layer_2's hidden size (4), and the output 
#   dimension produces the scalar output of layer_3 (1).
# ==============================================================================


# 1. Implement train_from_diagram based on the architecture diagram
def train_from_diagram(tells, strike, alpha, epochs, seed):
    """
    Trains a 3-layer deep neural network (layer_0 -> layer_1 -> layer_2 -> layer_3)
    using stochastic gradient descent based strictly on the provided diagram.
    """
    np.random.seed(seed)
    
    # Initialize three weight matrices with prescribed shapes: 2 * random - 1
    weights_0_1 = 2 * np.random.random((3, 8)) - 1  # (3, 8)
    weights_1_2 = 2 * np.random.random((8, 4)) - 1  # (8, 4)
    weights_2_3 = 2 * np.random.random((4, 1)) - 1  # (4, 1)

    error_history = []

    for epoch in range(epochs):
        total_error = 0.0

        # Four sensings (samples) per epoch
        for i in range(len(tells)):
            # --- FORWARD
            layer_0 = tells[i:i+1]                      # (1, 3)
            
            layer_1_input = layer_0 @ weights_0_1        # (1, 8)
            layer_1 = np.maximum(0, layer_1_input)      # ReLU activation -> (1, 8)
            
            layer_2_input = layer_1 @ weights_1_2        # (1, 4)
            layer_2 = np.maximum(0, layer_2_input)      # ReLU activation -> (1, 4)
            
            layer_3 = layer_2 @ weights_2_3              # No activation -> (1, 1)

            # --- COMPARE
            goal = strike[i:i+1]                         # (1, 1)
            error = (layer_3 - goal) ** 2               # (1, 1)
            total_error += float(np.sum(error))

            # --- BACKWARD
            layer_3_delta = layer_3 - goal               # (1, 1)
            
            layer_2_delta = (layer_3_delta @ weights_2_3.T) * (layer_2_input > 0)  # (1, 4)
            layer_1_delta = (layer_2_delta @ weights_1_2.T) * (layer_1_input > 0)  # (1, 8)

            # --- UPDATE
            weights_2_3 -= alpha * (layer_2.T @ layer_3_delta)  # (4, 1)
            weights_1_2 -= alpha * (layer_1.T @ layer_2_delta)  # (8, 4)
            weights_0_1 -= alpha * (layer_0.T @ layer_1_delta)  # (3, 8)

        error_history.append(total_error)

    return error_history, weights_0_1, weights_1_2, weights_2_3


if __name__ == "__main__":
    # Trial dataset from Week 5
    tells = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])
    
    strike = np.array([
        [1],
        [1],
        [0],
        [0]
    ])

    # Configuration for Part 3
    alpha = 0.1
    epochs = 150
    seed = 4

    # 2. Run train_from_diagram and print total error every 30 epochs
    print("--- Training Deep Network (Seed 4) ---")
    np.random.seed(seed)
    
    # We execute the training routine to get error history and final weights
    error_history, w01, w12, w23 = train_from_diagram(tells, strike, alpha, epochs, seed)

    for ep in range(epochs):
        if (ep + 1) % 30 == 0 or ep == 0:
            print(f"Epoch {ep + 1:3d} | Total Error: {error_history[ep]:.6f}")

    # 3. Compute and print the four final predictions vs goals
    print("\n--- Final Predictions vs Goals ---")
    for i in range(len(tells)):
        layer_0 = tells[i:i+1]
        layer_1 = np.maximum(0, layer_0 @ w01)
        layer_2 = np.maximum(0, layer_1 @ w12)
        prediction = layer_2 @ w23
        
        print(f"Input: {tells[i]} | Prediction: {prediction[0, 0]:.4f} | Goal: {strike[i, 0]}")