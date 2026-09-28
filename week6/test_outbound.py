import numpy as np
from part3_build_from_diagram import train_from_diagram
from part1_refactored_loop import train, one_step_with_shapes, tells, strike


def test_deeper_net_runs():
    tells = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    strike = np.array([
        [1],
        [1],
        [0],
        [0]
    ])

    epochs = 3

    results = train_from_diagram(
        tells,
        strike,
        alpha=0.1,
        epochs=epochs,
        seed=4
    )

    error_history = results[0]

    assert len(error_history) == epochs


def test_deeper_net_is_deterministic():
    tells = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    strike = np.array([
        [1],
        [1],
        [0],
        [0]
    ])

    first_results = train_from_diagram(
        tells,
        strike,
        alpha=0.1,
        epochs=3,
        seed=4
    )

    second_results = train_from_diagram(
        tells,
        strike,
        alpha=0.1,
        epochs=3,
        seed=4
    )

    first_error_history = first_results[0]
    second_error_history = second_results[0]

    np.testing.assert_allclose(
        first_error_history,
        second_error_history
    )

def test_deeper_net_converges():
    tells = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    strike = np.array([
        [1],
        [1],
        [0],
        [0]
    ])

    results = train_from_diagram(
        tells,
        strike,
        alpha=0.1,
        epochs=150,
        seed=4
    )

    error_history = results[0]
    weights_0_1 = results[1]
    weights_1_2 = results[2]
    weights_2_3 = results[3]

    assert error_history[-1] < 0.001


    # forward pass:
              #np.maximum(0, ...) = relu function
    layer_1 = np.maximum(0, tells @ weights_0_1)
    layer_2 = np.maximum(0, layer_1 @ weights_1_2)
    predictions = layer_2 @ weights_2_3

    predicted_classes = predictions > 0.5 # turns into true/false
    correct_classes = strike > 0.5

    assert np.array_equal(predicted_classes, correct_classes) # makes sure that the predicted_classes = results

def test_refactor_matches_week5():
    reference_error = 1.5055622665134864e-05

    results = train(tells, strike, alpha = 0.2, epochs = 60, hidden_size = 4, seed = 1)

    error_history = results[2]
    actual_error = error_history[-1]

    difference = abs(actual_error - reference_error)

    assert difference < 1e-9

def test_part1_shapes():
    layer_0 = tells[0:1]
    target = strike[0:1]

    weights_0_1 = np.zeros((3, 4))
    weights_1_2 = np.zeros((4, 1))

    shapes = one_step_with_shapes(
        layer_0,
        target,
        weights_0_1,
        weights_1_2
    )

    assert shapes["layer_0"] == (1, 3)
    assert shapes["layer_1"] == (1, 4)
    assert shapes["layer_2"] == (1, 1)
    assert shapes["layer_1_delta"] == (1, 4)
    assert shapes["layer_2_delta"] == (1, 1)
    assert shapes["weights_0_1"] == (3, 4)
    assert shapes["weights_1_2"] == (4, 1)



if __name__ == "__main__":
    tests = [
        test_deeper_net_runs,
        test_deeper_net_is_deterministic,
        test_deeper_net_converges,
        test_refactor_matches_week5,
        test_part1_shapes
    ]

    failures = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as error:
            failures += 1
            print(f"FAIL: {test.__name__} -- {error}")

    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)