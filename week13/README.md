# Week 13: The Forge

Build a small NumPy autograd framework that records forward operations and
automatically propagates gradients backward.

## File structure

| File | Responsibility |
| --- | --- |
| `tensor.py` | Parts 1-2: Tensor engine and operations; Part 5 activations |
| `optim.py` | Part 3: SGD optimizer |
| `layers.py` | Parts 3 and 5: layers and loss |
| `train_mlp.py` | Part 4: bare and Sequential MLP training |
| `train_deep.py` | Part 5: nonlinear network training |
| `test_forge.py` | Part 6: eight tests, including numeric gradient checks |
| `forge.ipynb` | Notebook assembly, plots, and Part 7 reading |

These files are placeholders. The Tensor, training, and tests are not
implemented yet. Part 1 and Part 2 share `tensor.py`; coordinate edits.

## Setup

No external dataset is needed; see [DATASETS.md](../DATASETS.md).
Use NumPy for the framework and Matplotlib for the notebook's loss curves.
Run this week's scripts from the `week13/` directory.

## Part 1 acceptance checks

- A value used twice accumulates both contributions: `b.grad.data` is
  `[2, 2, 2, 2, 2]` in the handout's first example.
- A reused intermediate value waits for both contributions before sending
  them backward: `x.grad.data` is `[2, 2, 2, 2, 2]` in the second example.
- Gradients are plain tensors, and each creator receives its own copy.

## Notebook checklist

Include explanations between code cells, all three loss curves, analytic
versus numeric gradient-check results, and a summary of final losses.
Finish with a 150-250 word cell titled `Holocron - Reading` and restart
and run all cells before the final commit.
