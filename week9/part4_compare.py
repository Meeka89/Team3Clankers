
from pathlib import Path
import matplotlib.pyplot as plt

from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, relu, relu_deriv, sigmoid, sigmoid_deriv
from part3_upgrade import train


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "mnist.npz"
    (X_train, y_train), (X_test, y_test) = load_mnist(data_path)

    X_val = X_train[1000:2000]
    Y_val = one_hot(y_train[1000:2000])

    X_train = X_train[:1000]
    Y_train = one_hot(y_train[:1000])

    _, _, tanh_hist = train( X_train, Y_train, X_val, Y_val, hidden_f=tanh, hidden_d=tanh_deriv )

    _, _, relu_hist = train( X_train, Y_train, X_val, Y_val, hidden_f=relu, hidden_d=relu_deriv)

    _, _, sigmoid_hist = train( X_train, Y_train, X_val, Y_val,  hidden_f=sigmoid, hidden_d=sigmoid_deriv )

    results = { "tanh": tanh_hist[-1], "ReLU": relu_hist[-1], "sigmoid": sigmoid_hist[-1] }

    print("\nFinal Validation Accuracy")
    print("-------------------------")
    for name, accuracy in results.items():
        print(f"{name:<10} {accuracy:.1%}")

    plt.bar(results.keys(), results.values())
    plt.xlabel("Hidden Activation")
    plt.ylabel("Final Validation Accuracy")
    plt.title("MNIST: Comparing Hidden Activations")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("compare_plot.png"), dpi=150)
    plt.show()
