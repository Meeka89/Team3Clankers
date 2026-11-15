#---------layers.py----------#
#-------Andrew_Martin--------#
import numpy as np

from tensor import Tensor


class Layer:
    #Base class: every layer keeps a list of the Tensors it learns.

    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    #A fully connected layer: output = input @ weight + bias.

    def __init__(self, n_in, n_out):
        super().__init__()
        # Scaled random start so the signal neither dies nor explodes.
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_out), autograd=True)

        # Register both so the optimizer can find and update them.
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        # bias is one row; expand copies it once per row of the batch.
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    #A stack of layers run one after another.

    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        # Each layer's output becomes the next layer's input.
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        # Gather every sub-layer's parameters into one flat list.
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params



# These need tanh() and sigmoid() on Tensor (added to tensor.py in Part 5).

class Tanh(Layer):
    """Squash each value into (-1, 1). No parameters."""

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    """Squash each value into (0, 1). No parameters."""

    def forward(self, input):
        return input.sigmoid()


class MSELoss(Layer):
    """Squared error summed over the batch (summed, not averaged)."""

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)
