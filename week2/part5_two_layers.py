# Written by Carter Owens

# 1. Using your vect_mat_mul from Part 4, build a neural_network(input, weights) that processes two layers in sequence.
# 2. Run it on all four sparring sensings using the matrices below.
# 3. Print the hidden-layer values and final predictions separately. Add a comment block answering:
#   - What does each hidden value represent? As written, these two layers could be collapsed into one — so 
#     what would have to change for the middle layer to buy you something a single layer cannot?
# 4. Add a # –- NumPy version –- section using np.array and .dot() for both layers. Pay attention 
#    to whether you need .T (transpose) to get the shapes right.
import sys
print(sys.executable)


blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

sensings = [[blade_angle[i],balance[i], breath[i]] for i in range(4)]

def vect_mat_mul(vect, matrix):

    output = []

    for i in range(len(matrix)):
        output.append(w_sum(vect, matrix[i]))

    return output


def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output

def neural_network(input, weights):
    """
    weights[0] = input --> hidden
    weights[1] = hidden -->prediction
    """
    hid = vect_mat_mul(input, weights[0])
    pred = vect_mat_mul(input, weights[1])
    return hid, pred

# ih = input --> hidden
#       angle balance breath
ih_wgt = [[0.1, 0.2, -0.1], # -> hid[0]
          [-0.1, 0.1, 0.9], # -> hid[1]
          [0.1, 0.4, 0.1]]  # -> hid[2]

# hp = hidden --> prediction
#          hid0 hid1 hid2
hp_wgt = [[0.3, 1.1, -0.3], # -> opens_left?
          [0.1, 0.2, 0.0],  # -> strikes_high?
          [0.0, 1.3, 0.1]]  # -> feints?

weights = [ih_wgt, hp_wgt]
# Expected for sensing 0 (rounded): hidden = [0.86, 0.295, 1.23]
# pred = [0.2135, 0.145, 0.5065]


for i, sensing in enumerate(sensings):
    hid, pred = neural_network(sensing, weights)
    print(f"Sparring Sensing {i}:  blade_angle={sensing[0]}, balance={sensing[1]}, breath={sensing[2]}")
    print(f"   Hidden: {hid}")
    print(f"   Prediction: {pred}")


# –- NumPy version –-

import numpy as np

blade_angle_np = np.array([8.5, 9.5, 9.9, 9.0])
balance_np = np.array([0.65, 0.80, 0.80, 0.90])
breath_np = np.array([1.2, 1.3, 0.5, 1.0])

sensings_np = [[blade_angle_np[i],balance_np[i], breath_np[i]] for i in range(4)]


ih_wgt_np = np.array([[0.1, 0.2, -0.1], 
                      [-0.1, 0.1, 0.9], 
                      [0.1, 0.4, 0.1]] ) 

#          hid0 hid1 hid2
hp_wgt_np = np.array([[0.3, 1.1, -0.3], # -> opens_left?
                      [0.1, 0.2, 0.0],  # -> strikes_high?
                      [0.0, 1.3, 0.1]])  # -> feints?

weights_np = [ih_wgt_np, hp_wgt_np]

def neural_network_np(input, weights):
    hid = input.dot(weights[0].T)
    pred = hid.dot(weights[1].T)
    return hid, pred


print("\n" + "=" * 62)
print("NUMPY")
print("=" * 62)

for i, sensing_np in enumerate(sensings_np):
    hid_np, pred_np = neural_network(sensing_np, weights_np)
    print(f"Sparring Sensing {i}:  blade_angle={sensing_np[0]}, balance={sensing_np[1]}, breath={sensing_np[2]}")
    print(f"   Hidden: {hid_np}")
    print(f"   Prediction: {pred_np}")

