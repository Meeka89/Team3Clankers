import numpy as np
from part3_build_from_diagram import train_from_diagram


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

def test_part1_shapes():
    return

test_deeper_net_runs()
print("First test passed")

test_deeper_net_is_deterministic()
print("Second test passed")

test_deeper_net_converges
print("Third test passed")

