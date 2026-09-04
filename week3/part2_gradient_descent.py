#-----part2_gradient_descent.py-----#
#----------Andrew_Martin------------#

import numpy as np

balance = [0.65, 0.80, 0.80, 0.90]
clean = [1, 1, 0, 1]
# train on sensing 0: input = 0.65, goal = 1
weight = 0.5 # starting weight

# calculate the error FIX prediction 
def squared_error(predection, goal):
    return (prediction - goal) ** 2


#  =====LOOP======                        *i is for iterations       
def gradient_descent(input, goal, weight, i):

    # should the empty list go in the loop?
    errors = []

    for i in range(i):
        prediction = input * weight
        # first sensing should be:
        # input: 0.65
        # weight = 0.5
        # pred = 0.65 * 0.5 = 0.325
        
        # calculate the error in the loop
        error = squared_error(prediction, goal)
        # prediction: 0.325
        # goal: 1
        # error: (0.325 - 1) ** 2

        #calculate how wrong the pred was
        delta = prediction - goal

        # calculate how much the weight should change(weight_delta)
        weight_delta = 0.1 * delta * input

        # update teh weight
        weight -= weight_delta

        # save the error from this iteration
        errors.append(error)

        #Print!!
        print(i + 1, error, prediction, weight)

    return errors 

# -----NUmPy version ----
# MAKE SURE THEY MATCH
# use numpy float64 values for everything
np_input = np.float64(0.65)
np_goal = np.float64(1.0)
np_weight = np.float64(0.5)

# Run GD using numpy values
np_errors = gradient_descent(np_input, np_goal, np_weight, 10) # run them 10 times

# print the errors from numpy
print("NumPy errors: ", np_errors)

# testing on two different sensings

# Train on sensing 0: input = 0.65, goal = 1
weight = 0.5
gradient_descent(balance[0], clean[0], weight, 10)

#Train on sensing 1: input = 0.80, goal = 1
weight = 0.5
gradient_descent(balance[1], clean[1], weight, 10)

#-----------------------------------------------------------------------------------
# Does the same starting weight work well for all sensings?
# 
# A: idk
#-----------------------------------------------------------------------------------
        

