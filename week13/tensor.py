"""Part 1: the autograd Tensor."""

import numpy as np


class Tensor:
    def __init__(self, data, autograd=False, creators=None,
                creation_op=None, id=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        self.id = id if id is not None else np.random.randint(0, 100000)
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        if creators is not None:
            for c in creators:
                c.children[self.id] = c.children.get(self.id, 0) + 1

    def all_children_accounted_for(self):
        return all(cnt == 0 for cnt in self.children.values())

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception("cannot backprop more than once")
            self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        assert grad.autograd == False # grads carry no grads

        if self.creators is not None and \
            (self.all_children_accounted_for() or grad_origin is None):
            if self.creation_op == "add":
                self.creators[0].backward(Tensor(self.grad.data), self)
                self.creators[1].backward(Tensor(self.grad.data), self)
            if self.creation_op == "sub":
                self.creators[0].backward(Tensor(self.grad.data), self)
                self.creators[1].backward(
                Tensor(self.grad.__neg__().data), self)
            if self.creation_op == "mul":
                self.creators[0].backward(self.grad * self.creators[1], self)
                self.creators[1].backward(self.grad * self.creators[0], self)
            if self.creation_op == "mm":
                c0, c1 = self.creators
                c0.backward(Tensor(self.grad.data.dot(c1.data.T)), self)
                c1.backward(Tensor(c0.data.T.dot(self.grad.data)), self)
            if self.creation_op == "transpose":
                self.creators[0].backward(self.grad.transpose(), self)
            if "sum" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim, ds), self)
            if "expand" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim), self)
            if self.creation_op == "neg":
                self.creators[0].backward(self.grad.__neg__(), self)
            if self.creation_op == "sigmoid":
                out = self.data # the activation's own output
                self.creators[0].backward(
                    Tensor(self.grad.data * (out * (1 - out))), self)
            if self.creation_op == "tanh":
                out = self.data
                self.creators[0].backward(
                    Tensor(self.grad.data * (1 - out * out)), self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True,
                          creators=[self, other], creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True,
                        creators=[self], creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, autograd=True,
                    creators=[self, other], creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd=True,
                        creators=[self, other], creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True,
                        creators=[self], creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(
        list(self.data.shape) + [copies]).transpose(trans_cmd)
        if self.autograd:
            return Tensor(new_data, autograd=True,
                creators=[self], creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True,
                    creators=[self], creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), autograd=True,
                creators=[self, x], creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), autograd=True,
                creators=[self], creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True,
                creators=[self], creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def __repr__(self):
        return str(self.data)


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
