import numpy as np
from data import load_mnist, one_hot
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate(X, Y, W_0_1, W_1_2):
    layer_1 = relu(X @ W_0_1)
    layer_2 = layer_1 @ W_1_2

    predictions = np.argmax(layer_2, axis=1)
    targets = np.argmax(Y, axis=1)

    return np.mean(predictions == targets)

def relu(x):
    return (x > 0) * x


def relu_d(x):
    return x > 0


def train(X_train, Y_train, X_val, Y_val, iterations=350):
    np.random.seed(1)

    alpha = 0.005
    hidden_size = 40

    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

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
        val_hist.append(evaluate(X_val, Y_val, W_0_1, W_1_2))

    return W_0_1, W_1_2, train_hist, val_hist


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    print("Training images:", X_train.shape)
    print("Training targets:", Y_train.shape)
    print("Validation images:", X_val.shape)
    print("Validation targets:", Y_val.shape)
    print("Pixel range:", X_train.min(), X_train.max())

    W_0_1, W_1_2, train_hist, val_hist = train(
        X_train, Y_train, X_val, Y_val
    )

    best_pass = np.argmax(val_hist) + 1

    print(f"Final training accuracy: {train_hist[-1]:.1%}")
    print(f"Final validation accuracy: {val_hist[-1]:.1%}")
    print(f"Best validation accuracy: {max(val_hist):.1%}")
    print(f"Best validation pass: {best_pass}")

    passes = np.arange(1, len(train_hist) + 1)

    plt.plot(passes, train_hist, label="Training")
    plt.plot(passes, val_hist, label="Validation")
    plt.xlabel("Training pass")
    plt.ylabel("Accuracy")
    plt.title("MNIST: Overfitting on 1,000 Training Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("curves_part1.png"))
    plt.show()