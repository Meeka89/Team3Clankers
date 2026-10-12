""" 
The centerpiece. Add dropout and watch the gap narrow.
1. Start from Part 1 and keep hidden_size = 40 and 350 iterations, so dropout is the only
change between the two runs. Write train_dropout(X_train, Y_train, X_val, Y_val,
iterations=350, dropout=True); with dropout=False it must reproduce your Part 1 run
exactly.

2. On every training step (every image), after computing layer_1, build a fresh dropout mask and
apply it:
mask = np.random.randint(2, size=layer_1.shape) # ~50% ones
layer_1 *= mask * 2 # zero half, double the rest
Doubling divides the survivors by the keep probability (1/0.5 = 2), so the next layer sees the
same expected signal in training as at evaluation; this is called inverted dropout.

3. On the backward pass, apply the same mask to delta_1 (delta_1 *= mask). Explain in a
comment why a silenced neuron must receive no gradient. (This week the line changes no
printed number — see Hints for why — but keep it.)

4. Put one training step (forward, backward and weight update for one image) in its own function
that takes the mask as an argument and updates the weights in place: dropout_step(layer_0,
target, W_0_1, W_1_2, mask, alpha). train_dropout draws the mask and calls it; your
Part 5 test 5 calls it with a mask it chose.

5. At evaluation time, use no dropout — the whole network, no mask, no doubling. (Your
evaluate from Part 1 already does this; do not change it.)

6. Record training accuracy (the running count, so mask on) and validation accuracy each iteration;
plot both to curves_part3.png.

7. In a printed summary, with and without dropout (run both, same seed), report: the final training
accuracy (running count), the training accuracy from evaluate() on the finished weights (mask
off), the best validation accuracy and its pass, and the final validation accuracy.
"""

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


def dropout_step(layer_0, target, W_0_1, W_1_2, mask, alpha):
    
    layer_1 = relu(layer_0 @ W_0_1)
    if mask is not None:
        layer_1 *= mask * 2
    layer_2 = layer_1 @ W_1_2

    correct = int(np.argmax(layer_2) == np.argmax(target))

    delta_2 = layer_2 - target
    delta_1 = (delta_2 @ W_1_2.T) * relu_d(layer_1)
    
    if mask is not None:
        delta_1 *= mask

    W_1_2 -= alpha * (layer_1.T @ delta_2)
    W_0_1 -= alpha * (layer_0.T @ delta_1)

    return correct



def train_dropout(X_train, Y_train, X_val, Y_val, iterations=350, dropout=True):
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
            
            if dropout:
                mask =  np.random.randint(2, size=(1, hidden_size))
                correct += dropout_step(layer_0, Y_train[i], W_0_1, W_1_2, mask, alpha) 
            else:
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

    iterations = 350

    print("Training images:", X_train.shape)
    print("Training targets:", Y_train.shape)
    print("Validation images:", X_val.shape)
    print("Validation targets:", Y_val.shape)
    print("Pixel range:", X_train.min(), X_train.max())

    results = {}
    
    for dropout in (True, False):
        name = "dropout" if dropout else "no dropout"
        W_0_1, W_1_2, train_hist, val_hist = train_dropout(
            X_train, Y_train, X_val, Y_val, iterations=iterations, dropout=dropout
        )

        # mask off
        train_eval = evaluate(X_train, Y_train, W_0_1, W_1_2)  
        best_pass = np.argmax(val_hist) + 1
        
        print(f"--- {name} ---")
        print(f"Final training accuracy (running, mask on): {train_hist[-1]:.1%}")
        print(f"Training accuracy via evaluate (mask off):  {train_eval:.1%}")
        print(f"Best validation accuracy: {max(val_hist):.1%} at pass {best_pass}")
        print(f"Final validation accuracy: {val_hist[-1]:.1%}")
        print(f"Gap (train running - final val): {(train_hist[-1] - val_hist[-1]) * 100:.1f} points")

        results[name] = (train_hist, val_hist)

    passes = np.arange(1, iterations + 1)
   

    for name, (train_hist, val_hist) in results.items():
        plt.plot(passes, train_hist, label=f"Training ({name})")
        plt.plot(passes, val_hist, label=f"Validation ({name})")
    plt.xlabel("Training pass")
    plt.ylabel("Accuracy")
    plt.title("MNIST: Dropout vs. no dropout on 1,000 Training Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("curves_part3.png"))
    plt.show()