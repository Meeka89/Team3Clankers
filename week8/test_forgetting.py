import contextlib
import copy
import inspect
import io
import sys

import numpy as np
from data import load_mnist, one_hot
from part1_memorize import train as train_p1
from part2_early_stopping import train_early_stopping, evaluate
from part3_dropout import train_dropout, dropout_step
from part4_train_of_formation import train_batched


def get_fast_data():
    (X_train_full, y_train_full), _ = load_mnist()

    X_train = X_train_full[:1000]
    Y_train = one_hot(y_train_full[:1000])
    X_val = X_train_full[1000:2000]
    Y_val = one_hot(y_train_full[1000:2000])

    return X_train, Y_train, X_val, Y_val


def test_overfitting_reproduces():
    X_train, Y_train, X_val, Y_val = get_fast_data()

    _, _, train_hist, val_hist = train_p1(
        X_train, Y_train, X_val, Y_val, iterations=45
    )

    final_train = train_hist[-1]
    final_val = val_hist[-1]
    gap = final_train - final_val

    assert final_train > 0.95, (
        f"Expected final training accuracy > 0.95 and got {final_train:.4f}"
    )
    assert gap >= 0.10, (
        f"Expected train validation gap >= 0.10 and got {gap:.4f}"
    )


def test_no_test_set_leakage():
    funcs = [train_p1, train_early_stopping, train_dropout, train_batched]

    for func in funcs:
        params = list(inspect.signature(func).parameters.keys())

        assert params[:4] == ["X_train", "Y_train", "X_val", "Y_val"], (
            f"{func.__name__} should begin with X_train, Y_train, X_val, Y_val; "
            f"got {params[:4]}"
        )
        assert not any("test" in p.lower() for p in params), (
            f"{func.__name__} has a test set parameter"
        )

        source = inspect.getsource(func)
        assert "X_test" not in source and "y_test" not in source, (
            f"{func.__name__} appears to reference test set data"
        )


def test_early_stopping_restores_best():
    X_train, Y_train, X_val, Y_val = get_fast_data()

    W_0_1, W_1_2, _, _, val_hist = train_early_stopping(
        X_train, Y_train, X_val, Y_val, iterations=50, patience=10
    )

    best_recorded_val = max(val_hist)
    restored_val = evaluate(X_val, Y_val, W_0_1, W_1_2)

    assert np.isclose(restored_val, best_recorded_val, rtol=0, atol=1e-9), (
        f"Returned weights score {restored_val:.10f}, but best recorded "
        f"validation accuracy was {best_recorded_val:.10f}"
    )


def test_dropout_narrows_gap():
    X_train, Y_train, X_val, Y_val = get_fast_data()

    _, _, train_hist_no_drop, val_hist_no_drop = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=40, dropout=False
    )
    _, _, train_hist_drop, val_hist_drop = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=40, dropout=True
    )

    gap_no_drop = train_hist_no_drop[-1] - val_hist_no_drop[-1]
    gap_drop = train_hist_drop[-1] - val_hist_drop[-1]

    assert gap_drop < gap_no_drop, (
        f"Expected dropout to narrow the final train-validation gap; "
        f"without dropout={gap_no_drop:.4f}, with dropout={gap_drop:.4f}"
    )


def test_dropout_zeroes_right_gradient():
    layer_0 = np.ones((1, 5))
    target = np.array([[0, 1]])
    W_0_1 = np.ones((5, 4))
    W_1_2 = np.ones((4, 2))
    W_0_1_orig = copy.deepcopy(W_0_1)

    mask = np.array([[0, 0, 1, 1]])

    dropout_step(layer_0, target, W_0_1, W_1_2, mask, alpha=0.1)

    assert np.allclose(W_0_1[:, 0], W_0_1_orig[:, 0]), (
        "Incoming weights for masked hidden unit 0 changed"
    )
    assert np.allclose(W_0_1[:, 1], W_0_1_orig[:, 1]), (
        "Incoming weights for masked hidden unit 1 changed"
    )
    assert not np.allclose(W_0_1[:, 2], W_0_1_orig[:, 2]), (
        "Incoming weights for active hidden unit 2 did not change"
    )
    assert not np.allclose(W_0_1[:, 3], W_0_1_orig[:, 3]), (
        "Incoming weights for active hidden unit 3 did not change"
    )


if __name__ == "__main__":
    tests = [
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
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

    main()
