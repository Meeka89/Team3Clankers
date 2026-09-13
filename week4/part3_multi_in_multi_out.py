from helpers import vect_mat_mul
import numpy as np

def outer_prod(deltas, inputs):
    output = []

    for i in range(len(deltas)):
        row = []

        for j in range(len(inputs)):
            row.append(deltas[i] * inputs[j])

        output.append(row)

    return output


def gradient_descent_full(input, weights, trues, alpha, iterations):
    # Copy each row so the original starting weights are not changed
    weights = [row.copy() for row in weights]

    error_history = []
    weight_history = []

    for iteration in range(iterations):

        # Make one prediction for each output
        pred = vect_mat_mul(input, weights)

        # Calculate how far each prediction is from its target
        deltas = []

        for i in range(len(pred)):
            deltas.append(pred[i] - trues[i])

        # Mean squared error across all three outputs
        error = 0

        for i in range(len(pred)):
            error += (pred[i] - trues[i]) ** 2

        error /= len(pred)

        # Each output delta is multiplied by each input
        # This creates a 3x3 matrix of weight deltas
        weight_deltas = outer_prod(deltas, input)

        # Update every weight
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] -= alpha * weight_deltas[i][j]

        error_history.append(error)

        # Save a copy of the entire weight matrix
        weight_history.append([row.copy() for row in weights])

        print(
            f"Iteration {iteration}: "
            f"pred={pred}, "
            f"error={error}, "
            f"weights={weights}"
        )

    return weights, error_history, weight_history


# --- NumPy version ---

def gradient_descent_full_numpy(input, weights, trues, alpha, iterations):
    input = np.array(input, dtype=float)
    weights = np.array(weights, dtype=float)
    trues = np.array(trues, dtype=float)

    error_history = []
    weight_history = []

    for iteration in range(iterations):

        # Same as vect_mat_mul()
        pred = np.dot(weights, input)

        # Prediction - target for all three outputs at once
        deltas = pred - trues

        # Mean squared error across the three outputs
        error = np.mean(np.square(deltas))

        # Creates the same 3x3 weight delta matrix as outer_prod()
        weight_deltas = np.outer(deltas, input)

        # Update every weight at once
        weights -= alpha * weight_deltas

        error_history.append(error)
        weight_history.append(weights.copy())

    return weights, error_history, weight_history



if __name__ == "__main__":
            # blade balance breath
    input = [8.5, 0.65, 1.2]

    weights = [
        [0.1, 0.1, -0.3],   # opens_left
        [0.1, 0.2, 0.0],    # strikes_high
        [0.0, 1.3, 0.1]     # feints
    ]

    trues = [0.0, 1.0, 0.0]

    alpha = 0.01
    iterations = 15

    final_weights, errors, weight_history = gradient_descent_full(
        input,
        weights,
        trues,
        alpha,
        iterations
    )

    print("\nFinal weight matrix:")

    for row in final_weights:
        print(row)

    numpy_weights, numpy_errors, numpy_weight_history = gradient_descent_full_numpy(
    input,
    weights,
    trues,
    alpha,
    iterations
    )

    print("\nNumPy final weight matrix:")
    print(numpy_weights)

    print(
        "\nFrom-scratch and NumPy match:",
        np.allclose(final_weights, numpy_weights, atol=1e-10)
    )   

# final weight matrix:
# [0.03634677021304352, 0.0951324000751151, -0.30898633832286443]
# [0.10229381008241284, 0.20017540900630212, 0.00032383201163475655]
# [-0.11067633647641982, 1.2915365154459209, 0.08437510543862306]

# Within each row, the blade_angle weight changed the most. This
# is because the input for the blade angle was the highest at 8.5. 
# Because weight_delta = delta * input, that high input made the
# weight delta higher than the other inputs' weight_deltas. 
# therefore the blade_angle weight changed the most.

# The feints row changed the most because it originally had the highest delta.
# the prediction was .965 while the target was 0.0

# The strikes_high row changed the least because its prediction started at
# 0.98 and the target was 1.0. Its delta was very small so all the weight
# deltas for that row were also small.
