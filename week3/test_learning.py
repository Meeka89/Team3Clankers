# test_learning.py -- Unit tests for the Week 3 learning loop
from part1_error import squared_error
#from part2_gradient_descent import gradient_descent
from part3_alpha import gradient_descent_alpha, gradient_descent_alpha_numpy
from part4_alpha_experiment import detect_divergence

def test_squared_error_basic():
    """Squared error of known values."""
    assert squared_error(1.0, 1.0) == 0.0, "Perfect prediction should give 0 error"
    assert squared_error(0.5, 1.0) == 0.25, "0.5 vs 1.0 should give 0.25"

def test_squared_error_negative():
    """Testing squared_error with negative values."""
    assert squared_error(-1.0, 1.0) == 4.0, "Squared error always be non-negative"
    assert squared_error(-2.0, -3.0) == 1.0

def test_gradient_descent_convergence_error_decreases():
    """Testing error decreasing over iterations."""

    errors = gradient_descent(0.65, 1.0, 0.5, 100)
    
    assert errors[0] > errors[-1], "Error should decrease over iterations"

def test_gradient_descent_convergence_final_error_small():
    """Testing final error being smalle"""

    errors = gradient_descent(0.65, 1.0, 0.5, 100)
    assert errors[-1] < 0.1, "Final error should be small"

def test_gradient_descent_alpha_converges():
    """Testing gradient descent with alpha converging."""

    errors = gradient_descent_alpha(2.0, 0.8, 0.5, 0.1, 20)

    assert errors[0] > errors[-1], "Error should decrease with alpha"
    assert errors[-1] < 0.01, "Final error should be small with alpha"

def test_gradient_descent_without_alpha_diverges():
    """Testing gradient descent without alpha diverging."""

    errors = gradient_descent(2.0, 0.8, 0.5, 20)

    assert errors[0] < errors[-1], "Error should increase without alpha"
    assert errors[-1] > errors[0], "Error blows up without alpha"

def test_detect_divergence():
    """Divergence detection which correctly flags growing error sequence and correctly returns False for shrinking one."""

    growing_errors = [0.04, 0.16, 0.64, 2.56, 10.24, 40.96, 163.84, 655.36]
    assert detect_divergence(growing_errors) == True, "Should flag growing error sequence as diverged"
    
    
    shrinking_errors = [2.1225, 1.8903, 1.6834, 1.4999, 1.3362, 1.1902, 1.0597, 0.1753]
    assert detect_divergence(shrinking_errors) == False, "Should return False for shrinking error sequence"

def test_gradient_descent_alpha_numpy_agreement():
    """From-scratch and NumPy produce same result within floating-point tolerance."""
    scratch_errors = gradient_descent_alpha(2.0, 0.8, 0.5, 0.1, 20)
    numpy_errors = gradient_descent_alpha_numpy(2.0, 0.8, 0.5, 0.1, 20)
    
    for i in range(len(scratch_errors)):
        a = scratch_errors[i]
        b = numpy_errors[i]
        assert abs(a - b) < 1e-10


if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f'PASS: {test_name}')
        except AssertionError as e:
            print(f'FAIL: {test_name} -- {e}')