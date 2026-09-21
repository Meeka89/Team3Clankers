import contextlib
import io
import numpy as np

from part1_single_layer_fails import single_layer_train, tells, strike
from part2_forward_hidden import relu, forward
from part3_one_backprop_step import relu2deriv, one_step
from part4_full_training_loop import train


def initial_weights():
    np.random.seed(1)
    return 2 * np.random.random((3, 4)) - 1, 2 * np.random.random((4, 1)) - 1


def test_relu():
    assert np.array_equal(relu(np.array([-1, 0, 1, 2])), [0, 0, 1, 2])


def test_relu_derivative():
    values = np.array([-1, 0, 1, 2])
    result = relu2deriv(values)
    assert result.shape == values.shape
    assert np.array_equal(result, [0, 0, 1, 1])


def test_single_layer_fails():
    # Part 1 prints its error instead of returning it.
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        single_layer_train(tells, strike.ravel(), 0.1, 60, 1)
    errors = [line for line in output.getvalue().splitlines() if "error:" in line]
    final_error = float(errors[-1].split("error:")[1])
    assert final_error > 0.5


def test_hidden_shape():
    w01, w12 = initial_weights()
    hidden, _ = forward(tells[0:1], w01, w12)
    assert hidden.shape == (1, 4)


def test_output_shape():
    w01, w12 = initial_weights()
    _, prediction = forward(tells[0:1], w01, w12)
    assert prediction.shape == (1, 1)


def test_one_step_reduces_error():
    w01, w12 = initial_weights()
    _, before = forward(tells[0:1], w01, w12)
    w01, w12, _, _ = one_step(tells[0:1], strike[0:1], w01, w12, 0.2)
    _, after = forward(tells[0:1], w01, w12)
    assert np.sum((after - strike[0:1]) ** 2) < np.sum((before - strike[0:1]) ** 2)


def test_training_converges():
    w01, w12, errors = train(tells, strike, 0.2, 60, 4, 1)
    _, predictions = forward(tells, w01, w12)
    assert len(errors) == 60
    assert errors[-1] < 0.01
    assert np.sum((predictions - strike) ** 2) < 0.01
    assert np.array_equal(predictions > 0.5, strike > 0.5)


def test_training_is_deterministic():
    first = train(tells, strike, 0.2, 60, 4, 1)
    second = train(tells, strike, 0.2, 60, 4, 1)
    assert np.allclose(first[0], second[0], atol=1e-10, rtol=0)
    assert np.allclose(first[1], second[1], atol=1e-10, rtol=0)


if __name__ == "__main__":
    tests = [(name, value) for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    failures = 0
    for name, test in sorted(tests):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                test()
            print(f"PASS: {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL: {name} -- {error}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
