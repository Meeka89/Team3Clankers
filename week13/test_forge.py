"""Part 6: unit tests and numeric gradient checks for The Forge."""

import sys

import numpy as np

from tensor import Tensor
from layers import Linear, Sequential, Tanh, Sigmoid, MSELoss
from optim import SGD


def test_multi_use_accumulation():
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)
    d = a + b
    e = b + c
    (d + e).backward(Tensor([1, 1, 1, 1, 1]))
    np.testing.assert_allclose(b.grad.data, [2, 2, 2, 2, 2])


def test_add_passes_gradient():
    a = Tensor([1, 2], autograd=True)
    b = Tensor([3, 4], autograd=True)
    c = Tensor([5, 6], autograd=True)
    d = Tensor([7, 8], autograd=True)
    loss = (a + b) + (c + d)
    loss.backward(Tensor([1, 1]))
    for value in (a, b, c, d):
        np.testing.assert_allclose(value.grad.data, [1, 1])


def test_numeric_gradient_mul_sum():
    x = Tensor([2.0, 3.0], autograd=True)
    y = (x * x).sum(0)
    y.backward()
    analytic = x.grad.data.copy()

    eps = 1e-5
    numeric = np.zeros_like(x.data, dtype=float)
    for i in range(len(x.data)):
        plus = x.data.copy()
        minus = x.data.copy()
        plus[i] += eps
        minus[i] -= eps
        f_plus = np.square(plus).sum()
        f_minus = np.square(minus).sum()
        numeric[i] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, [4.0, 6.0], atol=1e-8)
    np.testing.assert_allclose(analytic, numeric, atol=1e-4)


def test_sequential_loss_decreases_each_step():
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = Tensor([[0], [1], [0], [1]], autograd=True)
    model = Sequential([Linear(2, 3), Linear(3, 1)])
    optimizer = SGD(parameters=model.get_parameters(), alpha=0.01)

    losses = []
    for _ in range(10):
        pred = model.forward(data)
        loss = ((pred - target) * (pred - target)).sum(0)
        losses.append(float(loss.data[0]))
        loss.backward()
        optimizer.step()

    assert all(after < before for before, after in zip(losses, losses[1:])), losses


def test_nonlinear_model_has_four_parameters():
    model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
    assert len(model.get_parameters()) == 4
    assert all(isinstance(p, Tensor) for p in model.get_parameters())


def test_shared_intermediate_waits_for_all_children():
    x = Tensor([1, 2, 3, 4, 5], autograd=True)
    y = Tensor([1, 1, 1, 1, 1], autograd=True)
    s = x + y
    a = Tensor([2, 2, 2, 2, 2], autograd=True)
    b = Tensor([3, 3, 3, 3, 3], autograd=True)
    loss = (s + a) + (s + b)
    loss.backward(Tensor([1, 1, 1, 1, 1]))
    np.testing.assert_allclose(x.grad.data, [2, 2, 2, 2, 2])


def test_mm_reuse_accumulates_correctly():
    X = Tensor([[1., 2.], [3., 4.]], autograd=True)
    W = Tensor([[0.5, -0.5], [1., 0.25]], autograd=True)
    A = Tensor([[1., 0.], [2., 1.]], autograd=True)
    B = Tensor([[0., 1.], [1., 3.]], autograd=True)
    h = X.mm(W)
    (h.mm(A) + h.mm(B)).sum(0).sum(0).backward()
    expected = [[8.0, 28.0], [12.0, 42.0]]
    np.testing.assert_allclose(W.grad.data, expected)


def test_whole_nonlinear_network_gradient():
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = Tensor([[0], [1], [0], [1]], autograd=True)
    model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
    criterion = MSELoss()

    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backward()
    parameter = model.get_parameters()[0]
    index = (0, 0)
    analytic = float(parameter.grad.data[index])

    original = float(parameter.data[index])
    eps = 1e-5

    parameter.data[index] = original + eps
    plus = float(criterion.forward(model.forward(data), target).data[0])

    parameter.data[index] = original - eps
    minus = float(criterion.forward(model.forward(data), target).data[0])

    parameter.data[index] = original
    numeric = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    tests = [
        ("Multi-use gradient accumulation", test_multi_use_accumulation),
        ("Addition passes gradients", test_add_passes_gradient),
        ("Numeric gradient check", test_numeric_gradient_mul_sum),
        ("Sequential model learns", test_sequential_loss_decreases_each_step),
        ("Nonlinear model parameters", test_nonlinear_model_has_four_parameters),
        ("Shared intermediate gradients", test_shared_intermediate_waits_for_all_children),
        ("Matrix multiplication gradient", test_mm_reuse_accumulates_correctly),
        ("Whole network gradient check", test_whole_nonlinear_network_gradient),
    ]

    print("=" * 60)
    print("RUNNING THE FORGE UNIT TEST SUITE")
    print("=" * 60)

    passed_count = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed_count += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    print("-" * 60)
    print(f"Results: {passed_count}/{len(tests)} tests passed.")
    print("=" * 60)

    if passed_count < len(tests):
        sys.exit(1)
