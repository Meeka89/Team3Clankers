import numpy as np
from tensor import Tensor


class Layer(object):
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        self.use_bias = bias

        W = (np.random.rand(n_inputs, n_outputs) - 0.5) * 0.1
        self.weight = Tensor(W, autograd=True)
        self.parameters.append(self.weight)

        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)


class Sequential(Layer):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = []
        for l in self.layers:
            params += l.get_parameters()
        return params


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


class MSELoss(object):
    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        W = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(W, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)


class CrossEntropyLoss(object):
    def forward(self, input, target):
        return input.cross_entropy(target)