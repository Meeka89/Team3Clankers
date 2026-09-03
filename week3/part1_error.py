
"""
By Carter Owens

Assignment:
1. Write squared_error(prediction, goal) from scratch (no imports) returning (prediction - goal) ** 2.

2. Write mean_squared_error(predictions, goals) that averages the squared errors across a list of predictions.

3. Add a # –- NumPy version –- section. Rewrite both functions using np.square and np.mean.
Comment each line explaining what NumPy does compared to your from-scratch version.

4. Verify both versions agree for all four sensings using abs(a - b) < 1e-10 rather than ==. 
Print the results side by side. 
In a comment, note whether they happen to match exactly here — and why that is not something to lean on: 
NumPy is free to add up a sum’s terms in a different order than your loop does, and a different order can change the last bit.
"""

# Simple squared error function from scratch
def squared_error(prediction, goal):
    return (prediction - goal) ** 2

# from-scratch mean squared error
def mean_squared_error(predictions, goals):
    squared_errors = 0
    for i in range(len(predictions)):
        squared_errors += squared_error(predictions[i], goals[i])
    return squared_errors / len(predictions)



# values given and empty lists set up
balance = [0.65, 0.80, 0.80, 0.90]
clean = [1, 1, 0, 1]
weight = 0.5
predictions = []
errors = []

# loop for calculating errors 
# compares prediction and clean for each element
for i in range(len(balance)):        
    pred = balance[i] * weight
    predictions.append(pred)
    errors.append(squared_error(pred, clean[i]))
 
mse = mean_squared_error(predictions, clean)

# --- Numpy Version ---

import numpy as np

# numpy squared error
def squared_error_np(pred, goal):
    # np.square squares each element in teh array
    # uses this instead of squaring each (pred -goal) one at a time
    # (more important for mse but still)
    return np.square(pred - goal)

# numpy mean squared error
def mean_squared_error_np(preds, goals):
   # np.square is used again to find the squared errors in one call
   # np.mean averages each of the squared errors (pred - goal) across the array at once
   # Naturally, combining np.mean and np.square allows us to copmute MSE in one line
   return np.mean(np.square(preds - goals))

# numpy values and lists
balance_np = np.array([0.65, 0.80, 0.80, 0.90])
clean_np = np.array([1, 1, 0, 1])
weight = 0.5
predictions_np = []
errors_np = []

# numpy loop for find the errors
for i in range(len(balance)):        
    pred_np = balance_np[i] * weight
    predictions_np.append(pred_np)
    errors_np.append(squared_error_np(pred_np, clean_np[i]))

# numpy creation of the predictions and in turn the MSE
predictions_np = np.array(predictions_np)
mse_np = mean_squared_error_np(predictions_np, clean_np)


# Comparisons of the from-scratch and the Numpy
# and why we use `abs(a -b) < 1e-10``

# All the four errors and the final MSE match exactly here because it
# is a tiny dataset. 
# NumPys np.mean is free to add up a sum’s terms in a different
# order than your loop does, and a different order can change the last bit.
# Which means in a different dataset, the two implementations could differ slightly 
# from each other
# So, we are forced to check with abs(a -b) < 1e-10 instead of an easy ==

print("Squared errors (scratch vs numpy):")
for i in range(len(errors)):
    match = abs(errors[i] - errors_np[i]) < 1e-10
    print(f"  Element {i}: scratch={errors[i]:.6f}  numpy={errors_np[i]:.6f}  match={match}")

print(f"\nMSE: scratch={mse:.6f}  numpy={mse_np:.6f}  match={abs(mse - mse_np) < 1e-10}")

