# week4/test_multi.py -- Unit tests for Week 4 multi-input/output gradient descent
import numpy as np
from helpers import normalize
from part1_multi_input import gradient_descent_multi
from part2_multi_output import gradient_descent_outputs
from part3_multi_in_multi_out import gradient_descent_full, gradient_descent_full_numpy, outer_prod
from part4_freeze import gradient_descent_frozen


# 1. Normalization
def test_normalize_scales_and_preserves_input():
    """Verify normalize leaves max reading at 1.0, scales rest, and does not mutate input."""
    original = [2.0, 4.0, 8.0]
    original_copy = list(original)

    result = normalize(original)

    assert result == [0.25, 0.5, 1.0], f"Expected [0.25, 0.5, 1.0], got {result}"
    assert original == original_copy, "normalize modified the original list"


# 2. Multi-Input GD
def test_multi_input_gd_reduces_error():
    """Verify gradient_descent_multi reduces error over iterations."""
    sensing = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1.0
    alpha = 0.001
    iterations = 10

    _, error_history, _ = gradient_descent_multi(
        sensing, weights, true, alpha, iterations
    )

    assert error_history[-1] < error_history[0], (
        f"Final error ({error_history[-1]:.6f}) should be less than initial ({error_history[0]:.6f})"
    )


def test_multi_input_gd_prediction_close_to_goal():
    """Verify final prediction is close to goal for an easy sensing input."""
    sensing = [1.0, 1.0, 1.0]
    weights = [0.1, 0.1, 0.1]
    true = 1.0
    alpha = 0.1
    iterations = 50

    final_w, _, _ = gradient_descent_multi(
        sensing, weights, true, alpha, iterations
    )
    final_pred = sum(s * w for s, w in zip(sensing, final_w))

    assert abs(final_pred - true) < 1e-3, (
        f"Final prediction ({final_pred:.4f}) is not close to target ({true})"
    )


# 3. Multi-Output GD
def test_multi_output_gd_predictions_move_toward_target():
    """Verify each output's prediction moves closer to its target."""
    input_val = 0.65
    weights = [0.3, 0.2, 0.9]
    trues = [0.0, 1.0, 0.0]
    alpha = 0.1
    iterations = 20

    initial_errors = [abs(w * input_val - t) for w, t in zip(weights, trues)]
    final_w, _, _ = gradient_descent_outputs(
        input_val, weights, trues, alpha, iterations
    )
    final_errors = [abs(w * input_val - t) for w, t in zip(final_w, trues)]

    for i in range(len(trues)):
        assert final_errors[i] < initial_errors[i], (
            f"Output {i} error did not decrease: initial {initial_errors[i]}, final {final_errors[i]}"
        )


def test_multi_output_gd_weights_change_unless_pred_equals_target():
    """Verify all weights change unless prediction equals target (zero delta)."""
    input_val = 2.0
    # w0 * input_val = 0.5 * 2.0 = 1.0 -> target = 1.0 (zero delta)
    weights = [0.5, 0.2, 0.9]
    trues = [1.0, 0.0, 0.0]
    alpha = 0.05
    iterations = 5

    final_w, _, _ = gradient_descent_outputs(
        input_val, weights, trues, alpha, iterations
    )

    assert abs(final_w[0] - weights[0]) < 1e-12, "Weight 0 changed despite zero delta!"
    assert abs(final_w[1] - weights[1]) > 1e-6, "Weight 1 should have changed."
    assert abs(final_w[2] - weights[2]) > 1e-6, "Weight 2 should have changed."


# 4. Outer Product
def test_outer_product_matrix():
    """Verify outer_prod([1, 2], [3, 4, 5]) produces expected 2x3 matrix."""
    result = outer_prod([1, 2], [3, 4, 5])
    expected = [[3, 4, 5], [6, 8, 10]]
    assert result == expected, f"Expected {expected}, got {result}"


# 5. Freezing
from part4_freeze import gradient_descent_frozen  # Updated name


def test_freezing_frozen_weights_do_not_change():
    """Verify weight indices in frozen remain unchanged across iterations."""
    sensing = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1.0
    alpha = 0.01
    iterations = 10
    frozen = [0, 2]  # Freeze 1st and 3rd weights

    final_w, _, _ = gradient_descent_frozen(
        sensing, weights, true, alpha, iterations, frozen
    )

    assert final_w[0] == weights[0], "Frozen weight at index 0 changed!"
    assert final_w[2] == weights[2], "Frozen weight at index 2 changed!"


def test_freezing_unfrozen_weights_still_change():
    """Verify weights NOT in frozen continue to update."""
    sensing = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1.0
    alpha = 0.01
    iterations = 10
    frozen = [0, 2]  # Index 1 is unfrozen

    final_w, _, _ = gradient_descent_frozen(
        sensing, weights, true, alpha, iterations, frozen
    )

    assert abs(final_w[1] - weights[1]) > 1e-6, "Unfrozen weight at index 1 did not change!"

# 6. From-Scratch vs. NumPy Agreement
def test_from_scratch_vs_numpy_agreement():
    """Verify full GD pure-Python and NumPy implementations match within 1e-10."""
    inputs = [8.5, 0.65, 1.2]
    weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]
    trues = [0.0, 1.0, 0.0]
    alpha = 0.01
    iterations = 15

    py_w, py_err, _ = gradient_descent_full(inputs, weights, trues, alpha, iterations)
    np_w, np_err, _ = gradient_descent_full_numpy(inputs, weights, trues, alpha, iterations)

    for r_py, r_np in zip(py_w, np_w):
        for w_py, w_np in zip(r_py, r_np):
            assert abs(w_py - w_np) < 1e-10, f"Weight mismatch: {w_py} vs {w_np}"

    for e_py, e_np in zip(py_err, np_err):
        assert abs(e_py - e_np) < 1e-10, f"MSE mismatch: {e_py} vs {e_np}"


# Test Runner
if __name__ == "__main__":
    tests = [
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f"PASS: {test_name}")
        except AssertionError as e:
            print(f"FAIL: {test_name} -- {e}")