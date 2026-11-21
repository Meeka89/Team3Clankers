#---------test_week14.py----------#
#---------Andrew_Martin-----------#
"""Part 6: unit tests for Week 14. Run with:  python test_week14.py"""

import os
import traceback

import numpy as np

from tensor import Tensor
from layers import Sequential, Embedding, Tanh, Linear, LSTMCell, CrossEntropyLoss
from optim import SGD
from exploding_gradients import vanishing, exploding, lstm_cell_state_horizon

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "qa_oldtongue.txt")


# ---------------------------------------------------------------- helpers

def lstm_sequence_loss(embedding, model, xs, ys):
    """Run the LSTM over a (steps, batch) chunk and return the summed loss."""
    loss_fn = CrossEntropyLoss()
    hidden = model.init_hidden(xs.shape[1])
    total = None
    for t in range(xs.shape[0]):
        output, hidden = model.forward(embedding.forward(Tensor(xs[t])), hidden)
        loss = loss_fn.forward(output, Tensor(ys[t]))
        total = loss if total is None else total + loss
    return total


# ------------------------------------------------------------------ tests

def test_index_select_backward():
    """Rows selected [0,1,2,2,1] times: a repeated index must accumulate."""
    x = Tensor(np.eye(5), autograd=True)
    y = x.index_select(Tensor([[1, 2, 3], [2, 3, 4]]))
    assert y.data.shape == (2, 3, 5)
    y.backward()
    # Each selection adds a row of ones, so any column counts the selections.
    assert np.array_equal(x.grad.data[:, 0], [0, 1, 2, 2, 1])


def test_cross_entropy_acceptance():
    """Embedding(3,3) -> Tanh -> Linear(3,4) falls 1.3885... -> 0.1652... in 10 steps."""
    np.random.seed(0)
    data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
    target = Tensor(np.array([0, 1, 0, 1]), autograd=True)
    model = Sequential([Embedding(3, 3), Tanh(), Linear(3, 4)])
    criterion = CrossEntropyLoss()
    optim = SGD(model.get_parameters(), alpha=0.4)
    losses = []
    for _ in range(10):
        loss = criterion.forward(model.forward(data), target)
        loss.backward(Tensor(np.ones_like(loss.data)))
        optim.step()
        losses.append(float(loss.data))
    assert np.isclose(losses[0], 1.3885032434928422, rtol=0, atol=1e-12)
    assert np.isclose(losses[-1], 0.16527389263866668, rtol=0, atol=1e-12)


def test_untrained_perplexity():
    """Output-zeroed batch-1 perplexity equals the vocab size (62)."""
    assert os.path.exists(DATA_PATH), \
        "missing data/qa_oldtongue.txt -- see DATASETS.md for the download"
    text = open(DATA_PATH, encoding="utf-8").read()
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    data = np.array([char_to_idx[ch] for ch in text])

    batch_size, bptt = 16, 25
    data = data[:len(data) // batch_size * batch_size].reshape(batch_size, -1)

    np.random.seed(0)
    embedding = Embedding(len(vocab), 512)
    model = LSTMCell(512, 512, len(vocab))
    model.w_ho.weight.data *= 0

    xs = data[:, :bptt].T          # (steps, batch)
    ys = data[:, 1:bptt + 1].T
    loss = lstm_sequence_loss(embedding, model, xs, ys)
    perplexity = np.exp(float(loss.data) / bptt)
    assert len(vocab) == 62
    assert np.isclose(perplexity, 62), perplexity


def test_one_step_decreases_loss():
    """One SGD step lowers the LSTM's loss (and every parameter gets a gradient)."""
    np.random.seed(0)
    vocab, dim = 10, 8
    embedding = Embedding(vocab, dim)
    model = LSTMCell(dim, dim, vocab)
    params = embedding.get_parameters() + model.get_parameters()
    optim = SGD(params, alpha=0.1)
    xs = np.random.randint(0, vocab, (6, 4))
    ys = np.random.randint(0, vocab, (6, 4))

    before = lstm_sequence_loss(embedding, model, xs, ys)
    before.backward()
    # A cell that drops its forget gate leaves xf/hf with no gradient.
    for p in params:
        assert p.grad is not None
    optim.step()
    after = lstm_sequence_loss(embedding, model, xs, ys)
    assert float(after.data) < float(before.data)


def test_exploding_gradients():
    """exploding() is exactly 5**step; gated cell state stays bounded, ungated runs away."""
    grads = exploding()
    assert len(grads) == 10
    for step, g in enumerate(grads, start=1):
        assert np.max(np.abs(g)) == 5 ** step
    assert np.max(np.abs(grads[-1])) == 9765625

    # The vanishing half of the demo: ~0.0344 at step 1, ~2e-14 by step 10.
    v = [np.max(np.abs(g)) for g in vanishing()]
    assert np.isclose(v[0], 0.0344, atol=1e-4)
    assert v[-1] < 1e-13

    gated = lstm_cell_state_horizon(steps=50, forget_gate=True)
    ungated = lstm_cell_state_horizon(steps=50, forget_gate=False)
    assert np.isclose(gated[0], ungated[0])          # same writes at step 1
    assert gated[-1] < 1.5                            # ~0.96: bounded
    assert ungated[-1] > 3 * gated[-1]                # ~5.18: runs away
    long_gated = lstm_cell_state_horizon(steps=2000, forget_gate=True)
    long_ungated = lstm_cell_state_horizon(steps=2000, forget_gate=False)
    assert max(long_gated) < 5                        # no growth with the gate
    assert long_ungated[-1] > 20                      # ~34: no ceiling without it


def test_lstm_gradient_check():
    """Every LSTM gradient matches a central difference to about 1e-9."""
    np.random.seed(0)
    vocab, hidden, steps, batch = 5, 4, 4, 2
    embedding = Embedding(vocab, hidden)
    model = LSTMCell(hidden, hidden, vocab)
    params = embedding.get_parameters() + model.get_parameters()
    xs = np.random.randint(0, vocab, (steps, batch))
    ys = np.random.randint(0, vocab, (steps, batch))

    lstm_sequence_loss(embedding, model, xs, ys).backward()
    eps, worst = 1e-6, 0.0
    for p in params:
        analytic = p.grad.data.copy()
        for idx in np.ndindex(p.data.shape):
            old = p.data[idx]
            p.data[idx] = old + eps
            up = float(lstm_sequence_loss(embedding, model, xs, ys).data)
            p.data[idx] = old - eps
            down = float(lstm_sequence_loss(embedding, model, xs, ys).data)
            p.data[idx] = old
            worst = max(worst, abs((up - down) / (2 * eps) - analytic[idx]))
    assert worst < 1e-8, worst


# ------------------------------------------------------------------- main

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
