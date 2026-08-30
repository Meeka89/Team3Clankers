matrix = [
    [0.1, 0.1, -0.3], #opens left?
    [0.1, 0.2, 0.0], # strikes high?
    [0.0, 1.3, 0.1] # feints?
]

blade_angle = [8.5, 9.5, 9.9, 9.0]
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

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

for i in range(4):
    vect = []
    vect.append(blade_angle[i])
    vect.append(balance[i])
    vect.append(breath[i])

    prediction = vect_mat_mul(vect, matrix)

    print(prediction)

# ---NumPy version---
import numpy as np

matrix_np = np.array(matrix)

for i in range(4):
    vect = [blade_angle[i], balance[i], breath[i]]

    pred_scratch = vect_mat_mul(vect, matrix)

    vect_np = np.array(vect)
    prediction_np = vect_np.dot(matrix_np.T) # We use .T because NumPy's .dot calculates  the weighted sum using the columns of the
                                             # weight matrix per input instead of the rows.
                                             # .T flips the matrix along the diagonal axis so that way the 

    if i == 0:
        print("Input shape:", vect_np.shape) # the input shape is (3,) because each sensing has 3 features,
                                             # blade angle, balance, breath
        print("Weight matrix shape:", matrix_np.shape) # the weight matrix shape is (3,3) because there are 3 outputs,
                                                       # and each output has 3 weights, one weight for each input.

    print("Scratch:", pred_scratch)
    print("NumPy:", prediction_np)
    print("Match:", np.allclose(pred_scratch, prediction_np, atol=1e-9))