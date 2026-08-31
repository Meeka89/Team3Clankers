# week2/part3_multiple_outputs.py
import numpy as np

# Dataset (Sparring Impressions)
blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

# Weights for predictions: [opens_left?, strikes_high?, feints?]
weights = [0.3, 0.2, 0.9]

# ---------------------------------------------------------
# 1. Write ele_mul(scalar, vector) from scratch
# ---------------------------------------------------------
def ele_mul(scalar, vector):
    """Multiplies a single scalar by each element in a vector (list)."""
    return [scalar * w for w in vector]

# ---------------------------------------------------------
# 2. Write neural_network(input, weights)
# ---------------------------------------------------------
def neural_network(input_val, weights_list):
    """Takes a single input scalar and a vector of weights to return multiple predictions."""
    return ele_mul(input_val, weights_list)

# ---------------------------------------------------------
# 3. Run on all four balance values
# ---------------------------------------------------------
print("--- From Scratch Version ---")
for i, b in enumerate(balance):
    pred = neural_network(b, weights)
    print(f"Sensing {i} (balance = {b}): {pred}")

# ---------------------------------------------------------
# 4. NumPy version
# ---------------------------------------------------------
# NumPy utilizes broadcasting when multiplying a scalar by an ndarray.
# It automatically expands (broadcasts) the scalar across the array's 
# dimensions so the multiplication happens elementwise in optimized C code 
# without needing an explicit Python for-loop.

print("\n--- NumPy Version ---")
np_weights = np.array(weights)

for i, b in enumerate(balance):
    # Scalar multiplication with an np.array performs elementwise multiplication
    np_pred = b * np_weights
    print(f"Sensing {i} (balance = {b}): {np_pred}")