"""Train the Week 10 convolutional network on MNIST."""

from pathlib import Path

import numpy as np

from conv import conv_forward
from data import load_mnist, one_hot
from forms import softmax, tanh_deriv


def train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha):
    """Update both weight arrays in place and return the (9, 16) kernel gradient."""
    flat_patches, kernel_out, layer_1 = conv_forward(layer_0, kernels)
    # Save the activation's slope before dropout changes its values.
    slope = tanh_deriv(layer_1)
    layer_1 *= mask * 2
    layer_2 = softmax(layer_1.dot(W_1_2))

    # Average once over the actual batch size, including the short final batch.
    delta_2 = (layer_2 - Y_batch) / layer_0.shape[0]
    delta_1 = delta_2.dot(W_1_2.T) * slope
    delta_1 *= mask * 2

    # Compute delta_1 using the old classifier weights before updating them.
    W_1_2 -= alpha * layer_1.T.dot(delta_2)

    # Collect feedback from every image and position into the shared kernels.
    delta_1_reshaped = delta_1.reshape(kernel_out.shape)
    k_update = flat_patches.T.dot(delta_1_reshaped)
    kernels -= alpha * k_update
    return k_update


def evaluate(images, labels, kernels, W_1_2, batch_size=128):
    """Return accuracy without dropout, batching to limit memory use."""
    correct = 0
    for s in range(0, len(images), batch_size):
        layer_0 = images[s:s + batch_size].reshape(-1, 28, 28)
        _, _, layer_1 = conv_forward(layer_0, kernels)
        predictions = np.argmax(layer_1.dot(W_1_2), axis=1)
        targets = labels[s:s + batch_size]
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        correct += np.count_nonzero(predictions == targets)
    return correct / len(images)


def train(images, Y_train, val_images, val_labels,
          epochs=300, alpha=0.02, batch_size=128):
    """Train on the supplied split; return final weights and validation history."""
    np.random.seed(1)
    hidden_size = 676 * 16
    kernels = 0.02 * np.random.random((9, 16)) - 0.01
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1
    val_hist = []

    for epoch in range(1, epochs + 1):
        for s in range(0, len(images), batch_size):
            layer_0 = images[s:s + batch_size].reshape(-1, 28, 28)
            Y_batch = Y_train[s:s + batch_size]
            mask = np.random.randint(2, size=(len(layer_0), hidden_size))
            train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha)

        if epoch % 10 == 0 or epoch == epochs:
            accuracy = evaluate(val_images, val_labels, kernels, W_1_2)
            val_hist.append((epoch, accuracy))
            print(f"Epoch {epoch:3d}: validation accuracy {accuracy:.1%}", flush=True)

    return kernels, W_1_2, val_hist


def main():
    data_path = Path(__file__).resolve().parents[1] / "mnist.npz"
    (images, labels), (test_images, test_labels) = load_mnist(data_path)
    train_images = images[:1000]
    Y_train = one_hot(labels[:1000])
    val_images = images[1000:2000]
    val_labels = labels[1000:2000]

    kernels, W_1_2, val_hist = train(
        train_images, Y_train, val_images, val_labels
    )
    # Score the test set once, using the final weights, not a selected checkpoint.
    test_accuracy = evaluate(test_images, test_labels, kernels, W_1_2)
    print(f"Final test accuracy: {test_accuracy:.1%}")
    print(f"First-layer weights: {kernels.size:,}")
    print(f"Total weights: {kernels.size + W_1_2.size:,}")

    # Part 4 can load these final kernels to draw the 16 learned filters.
    output_path = Path(__file__).with_name("trained_weights.npz")
    np.savez(output_path, kernels=kernels, W_1_2=W_1_2,
             validation_history=np.asarray(val_hist), test_accuracy=test_accuracy)
    print(f"Saved final weights to {output_path.name}")


if __name__ == "__main__":
    main()
