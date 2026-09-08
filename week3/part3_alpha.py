import numpy as np

def gradient_descent(input, goal, weight, iterations):

    errors = []

    for iteration in range(iterations):
        prediction = input * weight
        error = (prediction - goal) ** 2
        delta = prediction - goal
        weight_delta = delta * input
        weight -= weight_delta

        errors.append(error)

        if iteration == 0:
            print("GRADIENT DESCENT NO ALPHA")

        print("Iteration:", iteration,
              "Prediction:", prediction,
              "Error:", error,
              "Weight", weight)
    return errors   
# the reason why the error blows up is because without alpha, the weight changes too much each iteration.
# The weight delta is large enought that the prediction goes past the goal. 
# the next prediction goes too far in the opposite direction and it get worse each time while flipping from
# too low to too high.

def gradient_descent_alpha(input, goal, weight, alpha, iterations):

    errors = []

    for iteration in range(iterations):
        prediction = input * weight
        error = (prediction - goal) ** 2
        delta = prediction - goal
        weight_delta = delta * input
        weight -= alpha * weight_delta

        errors.append(error)

        if iteration == 0:
            print("GRADIENT DESCENT WITH ALPHA")

        print("Iteration:", iteration,
              "Prediction:", prediction,
              "Error:", error,
              "Weight", weight)
    return errors   



# -- NumPy version --

import numpy as np


def gradient_descent_alpha_numpy(input, goal, weight, alpha, iterations):

    input = np.float64(input)
    goal = np.float64(goal)
    weight = np.float64(weight)
    alpha = np.float64(alpha)

    errors = []

    for i in range(iterations):
        prediction = input * weight
        error = np.square(prediction - goal)
        delta = prediction - goal
        weight_delta = delta * input

        weight -= alpha * weight_delta

        errors.append(error)

    return errors


if __name__ == "__main__":

    input = 2.0
    weight = 0.5
    goal = 0.8
    alpha = 0.1

    gradient_descent(input, goal, weight, 20)
    gradient_descent_alpha(input, goal, weight, alpha, 20)

    scratch_errors = gradient_descent_alpha(2.0, 0.8, 0.5, 0.1, 20)
    numpy_errors = gradient_descent_alpha_numpy(2.0, 0.8, 0.5, 0.1, 20)
    for a, b in zip(scratch_errors, numpy_errors):
        assert abs(a-b) < 1e-10

    blade_angle = [8.5, 9.5, 9.9, 9.0]
    clean = [1, 1, 0, 1]

    print("\n blade angle tests")
    for i in range(len(blade_angle)):
        print("\n round:", i)

        gradient_descent_alpha(
            blade_angle[i],
            clean[i],
            0.5, 
            0.1,
            20
        )
