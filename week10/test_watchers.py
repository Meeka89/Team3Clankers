
from pathlib import Path
import sys

import numpy as np

from conv import get_patches, conv_forward
from forms import softmax
from part3_train import train_step, train, evaluate
from data import load_mnist, one_hot


def test_patch_shape():
    layer = np.random.RandomState(0).rand(100, 28, 28)
    flat, _ = get_patches(layer)

    assert flat.shape == (67600, 9)
    print("PASS: patch shape")


def test_patch_content():
    layer = np.arange(28 * 28).reshape(1, 28, 28)
    flat, _ = get_patches(layer)

    expected = layer[0, 0:3, 0:3].flatten()
    assert np.array_equal(flat[0], expected)
    print("PASS: patch content")



def test_forward_shape():
    rng = np.random.RandomState(0)
    layer = rng.rand(3, 28, 28)
    kernels = 0.02 * rng.random((9, 16)) - 0.01

    _, _, layer_1 = conv_forward(layer, kernels)

    assert layer_1.shape == (3, 10816)
    assert np.all(layer_1 > -1) and np.all(layer_1 < 1)

    print("PASS: forward shape")




def test_training_step():
    rng = np.random.RandomState(0)
    layer_0 = rng.rand(3, 28, 28)
    kernels, W_1_2 = 0.5 * rng.randn(9, 16), 0.05 * rng.randn(10816, 10)
    Y = np.eye(10)[[3, 7, 1]]
    mask = rng.randint(2, size=(3, 10816))

    fp, es = get_patches(layer_0)
    ko = fp.dot(kernels)
    l1 = np.tanh(ko.reshape(layer_0.shape[0], 676*16))
    layer_2 = softmax((l1 * mask * 2).dot(W_1_2))
    delta_2 = (layer_2 - Y) / 3
    d1 = delta_2.dot(W_1_2.T) * (1 - l1**2) * mask * 2
    expected = fp.T.dot(d1.reshape(ko.shape))
    got = train_step(layer_0, Y, kernels.copy(), W_1_2.copy(), mask, alpha=0.02)
    assert got.shape == (9, 16)
    assert np.allclose(got, expected) or np.allclose(got, expected / 2)

    print("PASS: training step")



def test_learning():
    data_path = Path(__file__).resolve().parents[1] / "mnist.npz"
    (images, labels), _ = load_mnist(data_path)

    train_images = images[:1000]
    Y_train = one_hot(labels[:1000])
    val_images = images[1000:2000]
    val_labels = labels[1000:2000]

    kernels, W_1_2, _ = train(
        train_images,
        Y_train,
        val_images,
        val_labels,
        epochs=30,
        alpha=0.02,
        batch_size=128
    )

    accuracy = evaluate(val_images, val_labels, kernels, W_1_2)

    assert accuracy > 0.50
    print(f"PASS: learning ({accuracy:.1%} validation accuracy)")


if __name__ == "__main__":
    tests = [
        ("Patch shape", test_patch_shape),
        ("Patch content", test_patch_content),
        ("Forward shape", test_forward_shape),
        ("One training step", test_training_step),
        ("Network learns", test_learning),
    ]

    print("=" * 60)
    print("RUNNING WEEK 10 WATCHERS TEST SUITE")
    print("=" * 60)

    passed_count = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("-" * 60)
    print(f"Results: {passed_count}/{len(tests)} tests passed.")
    print("=" * 60)

    if passed_count < len(tests):
        sys.exit(1)

