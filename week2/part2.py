## multiple inputs, refer to assignment 3 for details ## done by saturday!!

"""
Name: Andrew Martin
Date: 8-27-26
Desc: Part 2 neuarl predictions 
"""
import numpy as np
# Todo:
#       
# w_sum(a, b)
# neural_network(input, weights)
# this will use w_sum to make a single prediction from three inputs
# Run all four sparring sensings
# use weighs: weights = [0.1, 0.2, 0.0]
#

#         angle, balance, breath
weights = [0.1, 0.2, 0.0]

blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]


def w_sum(a, b):
# make sure the lists contain the same number of values
    assert len(a) == len(b)

    total = 0      

    # multiply each input by its mathcing weight + add the products
    for i in range(len(a)):
        total += a[i] * b[i]

    return total

def neural_network(input, weights):
    # use the weigthed sum to combine the three inputs into one pred
    return w_sum(input, weights)

# rin the from scrathc version on all four sparring sensings
for i in range(4):
    input = [blade_angle[i], balance[i], breath[i]]
    prediction = neural_network(input, weights)
    print(f"sensing {i}: {prediction}")

# -- numpy version --

def neural_network_numpy(input, weights):
    # convert the input and weights list into numpy arrays
    input = np.array(input)
    weights = np.array(weights)

    # .dot() multiplies orresponding values and adds the results
    # doing the same weighted sum as w_sum() above
    return input.dot(weights)

for i in range(4):
    input = [blade_angle[i], balance[i], breath[i]]

    scratch_prediction = neural_network(input, weights)
    numpy_prediction = neural_network_numpy(input, weights)

    print(
        f"sensing {i}: "
        f"from scratch = {scratch_prediction}, "
        f"numPy = {numpy_prediction}, "
        f"match = {abs(scratch_prediction - numpy_prediction) < 1e-9}"
    )

    # with only 3 terms, the two versions should match exactly in this case.
    # with 10,000 terms, they might differ slightly in the final decimal place
    # because numpy may add the terms in a different order. this is why we use
    # a tolerance such as 1e-9 instead of comparing floating point values with ==