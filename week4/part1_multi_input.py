"""
Carter Owens


The case from Tuesday. Three inputs combine into a single prediction; one shared delta is dis-
tributed across three weights.


1. Write w_sum(a, b)        Check
and ele_mul(scalar,vector)  Check


2. Write gradient_descent_multi(input, weights, true, alpha, iterations) from scratch
that runs the full predict–compare–learn loop on one sensing and returns three things: the final
weights, the error history, and the weight history — the full weight vector at every iteration.
Part 5 plots the weight history, so record it now rather than changing this signature after the
PR is merged.


gradient_descent_multi(input, weights, true, alpha, iterations) check    


3. Train on sensing 0 ([8.5, 0.65, 1.2]) with starting weights [0.1, 0.2, -0.1], target true =
1.0, alpha = 0.01, and at least 10 iterations. Print a clean iteration log: iteration, prediction,
error, and the three current weights.


4. Add a # –- NumPy version –- section using np.dot and elementwise arithmetic. Verify the
NumPy output matches the from-scratch output within abs(a - b) < 1e-10.


5. Comment block: which weight changed the most over training? Which changed the least? Why?
"""
from helpers import w_sum, vect_mat_mul
import numpy as np
# import matplotlib




blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90] # stance reading, [0, 1]
breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second
clean = [1, 1, 0, 1] # ground truth: clean strike?


# trues[i] = [opens_left, strikes_high, feints] for sensing i
trues_multi = [[0.0, 1.0, 0.0], # sensing 0
               [1.0, 0.0, 0.0], # sensing 1
               [0.0, 0.0, 1.0], # sensing 2 (only feint)
               [1.0, 1.0, 0.0]] # sensing 3




def ele_mul(scalar, vector):
    # from scratch
    # multiply each entry of vector by scalar
    output = [0] * len(vector)
    for i in  range(len(vector)):
        output[i] = scalar * vector[i]


    return output


def gradient_descent_multi(input, weights, true, alpha, iterations):
    # from scratch
    # Returns the final weights, error history, and weight history
    error_history = []
    weight_history = [weights.copy()]
   
    for iter in range(iterations):
        # Predict
        pred = w_sum(input, weights)
        # Compare
        error = (pred - true) ** 2  # MSE
        delta = pred - true


        # Learn
        weight_deltas = ele_mul(delta, input)
        for i in range(len(weights)):
            weights[i] -= alpha * weight_deltas[i]


        weight_history.append(weights.copy())
        error_history.append(error)




    return weights, error_history, weight_history


if __name__ == '__main__':
    #   --- FROM SCRATCH Version ---
    print("\n")
    print("===== FROM SCRATCH VERSION =====")
    print("\n")


    blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
    balance = [0.65, 0.80, 0.80, 0.90] # stance reading, [0, 1]
    breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second
    clean = [1, 1, 0, 1] # ground truth: clean strike?


    sensings = [[blade_angle[i],balance[i], breath[i]] for i in range(4)]


    weights = [0.1, 0.2, -0.1]
    true = 1
    alpha = 0.01
    iterations = 10


    # trues[i] = [opens_left, strikes_high, feints] for sensing i
    trues_multi = [[0.0, 1.0, 0.0], # sensing 0
                   [1.0, 0.0, 0.0], # sensing 1
                   [0.0, 0.0, 1.0], # sensing 2 (only feint)
                   [1.0, 1.0, 0.0]] # sensing 3


    # conduct the gradient descent function which returns these variables
    final_weights, error_history, weight_history = gradient_descent_multi(sensings[0],
                                                                            weights,
                                                                            true,
                                                                            alpha,
                                                                            iterations)
 


    # Print iteration log
    for iter in range(iterations):
        #recalculate the prediction using our weight history
        pred = w_sum(sensings[0], weight_history[iter])
        print(f"Iteration {iter}: Pred - {pred:.6f}, Error - {error_history[iter]:.6f}")


    print(f"Current Weights: {[f'{w:.6f}' for w in final_weights]}")




    #   --- NumPy Version ---
    print("\n")
    print("===== NUMPY VERSION =====")
    print("\n")


    sensings_np = np.array(sensings)
    weights_np = np.array([0.1, 0.2, -0.1])


    for it_np in range(iterations):
        # PREDICT
        # np.dot replaces w_sum
        pred_np = np.dot(sensings_np[0], weights_np)
        # COMPARE
        error_np = (pred_np - true) ** 2
        delta = pred_np - true


        # LEARN
        # replaces the ele_mul and for-loop
        weights_np -= alpha * delta * sensings_np[0]


        print(f"Iter {it_np}: pred={pred_np:.4f} error={error_np:.6f}")


    print(f"Current Weights: {[f'{w:.6f}' for w in weights_np]}")
       
"""
5
Which weight changed the most over training?
Which changed the least? Why?


The first weight changed the most over training as it started
as 0.1, but changed to 0.116 after 10 iterations. On the other
end, the last weight changed from 0.2 to 0.201 after 10 iterations.


This is because the first input of sensing 0 (the blade angle) is 8.5,
which is much larger than the other inputs. The second input (balance)
is 0.65, which is by far the smallest. The sizes of these matter because
they factor into the weights and scale with each step.


1st weight - +0.016
2nd weight - +0.001
3rd weight - +0.003
"""
