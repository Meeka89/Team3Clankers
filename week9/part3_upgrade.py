from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax

def evaluate(X, Y, W_0_1, W_1_2, hidden_f=tanh):
    layer_1 = hidden_f(X @ W_0_1)
    scores = layer_1 @ W_1_2

    predictions = np.argmax(scores, axis=1)
    targets = np.argmax(Y, axis=1)

    return np.mean(predictions == targets)

def train_step(layer_0, Y_batch, W_0_1, W_1_2, mask, alpha, hidden_f=tanh, hidden_d=tanh_deriv):
    layer_1 = hidden_f(layer_0 @ W_0_1)
    slope = hidden_d(layer_1)

    layer_1 *= mask * 2
    layer_2 = softmax(layer_1 @ W_1_2)

    batch_size = layer_0.shape[0]
    delta_2 = (layer_2 - Y_batch) / batch_size

    delta_1 = (delta_2 @ W_1_2.T) * slope
    # Exact dropout gradient: double kept gradients, zero dropped gradients.
    delta_1 *= mask * 2

    W_1_2 -= alpha * (layer_1.T @ delta_2)
    W_0_1 -= alpha * (layer_0.T @ delta_1)

    return delta_1


def train(
    X_train, Y_train, X_val, Y_val, epochs=300, hidden_f=tanh, hidden_d=tanh_deriv):
    np.random.seed(1)

    alpha = 0.02
    hidden_size = 100
    batch_size = 100

    W_0_1 = 0.02 * np.random.random((784, hidden_size)) - 0.01
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    val_hist = []

    for i in range(epochs):
        for s in range(0, len(X_train), batch_size):
            layer_0 = X_train[s:s + batch_size]
            Y_batch = Y_train[s:s + batch_size]

            mask = np.random.randint(2, size=(len(layer_0), hidden_size))

            train_step(
                layer_0, Y_batch, W_0_1, W_1_2, mask,
                alpha, hidden_f, hidden_d
            )
        val_accuracy = evaluate(
            X_val, Y_val, W_0_1, W_1_2, hidden_f
        )
        val_hist.append(val_accuracy)
    return W_0_1, W_1_2, val_hist



if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "mnist.npz"
    (X_train, y_train), (X_test, y_test) = load_mnist(data_path)

    # Reserve validation images before shortening the training arrays.
    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])
    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])
    Y_test = one_hot(y_test)

    print("Training:", X_train.shape, Y_train.shape)
    print("Validation:", X_val.shape, Y_val.shape)
    print("Test:", X_test.shape, Y_test.shape)

    W_0_1, W_1_2, val_hist = train(
        X_train, Y_train, X_val, Y_val
    )

    print(f"Final validation accuracy: {val_hist[-1]:.1%}")

    epochs = np.arange(1, len(val_hist) + 1)

    plt.plot(epochs, val_hist, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST: tanh Hidden Layer + Softmax Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("upgrade_plot.png"))
    plt.show()

    test_accuracy = evaluate(X_test, Y_test, W_0_1, W_1_2)
    print(f"Final test accuracy: {test_accuracy:.1%}")

    # The final validation of Week8 batched was = 84.2%. 
    # Week 9 upgrade validation accuracy was 87.7%.
    # It improved 3.5 percentage points. Hidden size also increased from 40 to 100 neurons
    # so the gain in validation accuracy cannot just be attributed to the activation changes
    # Week 9 final test accuracy: 86.7%
