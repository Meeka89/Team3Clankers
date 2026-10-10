#------part2_early_stopping.py--------#
#----------Andrew_Martin--------------#
import copy
import numpy as np
from data import load_mnist, one_hot


def relu(x):
    return (x > 0) * x


def relu_d(x):
    return x > 0


def evaluate(X, Y, W_0_1, W_1_2):
    layer_1 = relu(X @ W_0_1)
    layer_2 = layer_1 @ W_1_2

    predictions = np.argmax(layer_2, axis=1)
    targets = np.argmax(Y, axis=1)

    return np.mean(predictions == targets)


def train_early_stopping(X_train, Y_train, X_val, Y_val, iterations=350, patience=10):
    np.random.seed(1)

    alpha = 0.005
    hidden_size = 40

    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

    best_val_acc = -1.0
    best_weights = (copy.deepcopy(W_0_1), copy.deepcopy(W_1_2))
    wait = 0
    stop_iter = iterations - 1

    for j in range(iterations):
        correct = 0
        for i in range(len(X_train)):
            layer_0 = X_train[i:i + 1]
            layer_1 = relu(layer_0 @ W_0_1)
            layer_2 = layer_1 @ W_1_2
            correct += int(np.argmax(layer_2) == np.argmax(Y_train[i]))
            delta_2 = layer_2 - Y_train[i:i + 1]
            delta_1 = (delta_2 @ W_1_2.T) * relu_d(layer_1)
            W_1_2 -= alpha * (layer_1.T @ delta_2)
            W_0_1 -= alpha * (layer_0.T @ delta_1)

        train_hist.append(correct / len(X_train))
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)
        val_hist.append(val_acc)

        # early stopping: remember the best weights, count passes with no improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = (copy.deepcopy(W_0_1), copy.deepcopy(W_1_2))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stop_iter = j
                break

    # outside the loop on purpose: a run that never hits patience still returns the best weights
    W_0_1, W_1_2 = best_weights

    return W_0_1, W_1_2, stop_iter, train_hist, val_hist


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    W_0_1, W_1_2, stop_iter, train_hist, val_hist = train_early_stopping(
        X_train, Y_train, X_val, Y_val, iterations=350, patience=10
    )

    best_val_acc = max(val_hist)
    best_pass = np.argmax(val_hist)

    print(f"Stopped at pass {stop_iter} (best pass {best_pass})")
    print(f"Best validation accuracy: {best_val_acc:.1%}")
    print(f"Restored weights score best validation exactly: "
          f"{evaluate(X_val, Y_val, W_0_1, W_1_2) == best_val_acc}")

    # the only test-set number in the whole assignment
    Y_test = one_hot(y_test)
    print(f"Test accuracy (restored weights): {evaluate(X_test, Y_test, W_0_1, W_1_2):.1%}")
