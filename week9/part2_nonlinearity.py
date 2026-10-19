# Carter Owens

from forms import relu
import numpy as np


if __name__ == "__main__":
    np.random.seed(1)

    # Step 1
    # Generate random inputs and weight matrices
    # with the respective shapes
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,3)
    W2 = np.random.randn(3,2)

    # Step 2: two linear layers
    Y_two = (X @ W1) @ W2

    # Step 3: one linear layer through colapsed matrix
    W_prime = W1 @ W2 
    Y_one = X @ W_prime

    # Step 4: show they are equal
    print(f"Two Linear Layer: {Y_two}")
    print(f"One Linear Layer: {Y_one}")
    print(f"Equal: {np.allclose(Y_two, Y_one)}")

    # Step 5: insert a form (relu) 
    Y_nl = relu(X @ W1) @ W2
    print(f"With relu (nonlinear): {Y_nl}")
    print(f"Nonlinear & Linear equal: {np.allclose(Y_nl, Y_one)}")

    """
    You can no longer collapse the matrix into one matrix because of the 
    added relu. Relu is not linear and the bend between layers stops it from 
    being able to collapse into one matrix. Because relu behaves differently 
    around 0 (it doesn't return negative), which causes it to be nonlinear 
    and causes it to not collapse.
    """

    # Best matrix (using least squares)
    W_best, _, _, _ = np.linalg.lstsq(X, Y_nl, rcond=None)
    Y_best = X @ W_best
    residual = Y_nl - Y_best

    # Show output
    # W_best -->  least squares solution
    # residual --> the difference between the best linear solution and the relu 
    print(f"Best matrix: {W_best}")

    # Checking if the best linear fit does reproduces the relu output.
    print(f"Best linear fit reproduces relu output: {np.allclose(Y_nl, Y_best)}")
    # The residual is nonzero showing that no linear solutions fits
    print(f"Residual (non-zero): {residual}")
    



