"""Part 1: the autograd Tensor."""

import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, creators=None,
                 creation_op=None, id=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.id = np.random.randint(0, 100000) if id is None else id
        self.children = {}

        # Tell each input that this result will owe it a gradient.
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 0
                creator.children[self.id] += 1

    def all_children_accounted_for(self):
        for count in self.children.values():
            if count != 0:
                return False
        return True

    def __add__(self, other):
        result = self.data + other.data

        # Record the operation only when both inputs track gradients.
        if self.autograd and other.autograd:
            return Tensor(result, autograd=True,
                          creators=[self, other], creation_op="add")
        return Tensor(result)

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        # The starting result gets a gradient of ones unless one is supplied.
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        # A child is reporting its contribution to this tensor's gradient.
        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception("cannot backprop more than once")
            self.children[grad_origin.id] -= 1

        # Keep contributions from every path instead of replacing them.
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        assert not grad.autograd, "Gradients must be plain tensors."

        # Wait for all children before sending the total farther backward.
        if self.creators is not None:
            if self.all_children_accounted_for() or grad_origin is None:
                if self.creation_op == "add":
                    for creator in self.creators:
                        # Each input gets its own independent gradient copy.
                        creator.backward(Tensor(self.grad.data), self)


if __name__ == "__main__":
    # Example 1: b is used twice, so its two contributions must add up.
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward(Tensor([1, 1, 1, 1, 1]))
    print("b gradient (expected all 2s):", b.grad.data)

    # Example 2: s must collect both contributions before passing them on.
    x = Tensor([1, 2, 3, 4, 5], autograd=True)
    y = Tensor([1, 1, 1, 1, 1], autograd=True)
    s = x + y
    loss = (s + a) + (s + c)
    loss.backward(Tensor([1, 1, 1, 1, 1]))
    print("x gradient (expected all 2s):", x.grad.data)
