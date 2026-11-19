#---------layers.py----------#
#-------Andrew_Martin--------#
import numpy as np

from tensor import Tensor


class Layer:
    # Base class: every layer keeps a list of the Tensors it learns.

    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    # A fully connected layer: output = input @ weight (+ bias).

    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        # Scaled random start so the signal neither dies nor explodes.
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.weight = Tensor(W, autograd=True)
        self.parameters.append(self.weight)

        # With bias=False, no bias Tensor is created at all (creating one
        # would also draw a Tensor id and shift the seeded random stream).
        self.use_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(n_out), autograd=True)
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.use_bias:
            # bias is one row; expand copies it once per row of the batch.
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)


class Sequential(Layer):
    # A stack of layers run one after another.

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
        for layer in self.layers:
            params += layer.get_parameters()
        return params


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


class Embedding(Layer):
    # A learnable lookup table: a weight you index instead of multiply.

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        W = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(W, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        # No gradient code: index_select's backward carries it.
        return self.weight.index_select(input)


class CrossEntropyLoss(object):
    # Thin wrapper; owns no parameters, so it is not a Layer.

    def forward(self, input, target):
        return input.cross_entropy(target)


class LSTMCell(Layer):
    # Three gates, a candidate, and a cell-state belt.

    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden)                # input -> gate (with bias)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)
        self.hf = Linear(n_hidden, n_hidden, bias=False)    # hidden -> gate (no bias)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)
        self.w_ho = Linear(n_hidden, n_output, bias=False)  # cell view -> logits

        # Register every sub-layer's parameters, in the order built above.
        for layer in [self.xf, self.xi, self.xo, self.xc,
                      self.hf, self.hi, self.ho, self.hc, self.w_ho]:
            self.parameters += layer.get_parameters()

    def forward(self, input, hidden):
        prev_hidden, prev_cell = hidden[0], hidden[1]
        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()
        c = (f * prev_cell) + (i * g)   # keep f of the old, write i of the new
        h = o * c.tanh()                # expose a gated view of the memory
        return self.w_ho.forward(h), (h, c)

    def init_hidden(self, batch_size=1):
        # Zeros, with feature 0 nudged to 1 to break symmetry (as Trask does).
        init_hidden = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_cell = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_hidden.data[:, 0] += 1
        init_cell.data[:, 0] += 1
        return (init_hidden, init_cell)
