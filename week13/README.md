# Week 13: The Forge

An autograd framework built from scratch in NumPy. A `Tensor` records every operation performed on
it, and `backward()` walks that record in reverse to hand each value its gradient. Write the forward
pass of any network built from these operations and the backward pass comes for free.

No dataset is needed this week. The toy problem is four rows, and the target is the second input bit.

## Files

| File | Part | What it does |
|---|---|---|
| `tensor.py` | 1, 2, 5 | The autograd `Tensor`: children bookkeeping, `backward()` with `+=` accumulation and the all-children wait, and the operation table (`add`, `neg`, `sub`, `mul`, `mm`, `transpose`, `sum`, `expand`, `tanh`, `sigmoid`), each with its backward branch |
| `optim.py` | 3 | `SGD`: `step()` updates each parameter and zeroes its gradient |
| `layers.py` | 3, 5 | `Layer`, `Linear`, `Sequential`, `Tanh`, `Sigmoid`, `MSELoss` |
| `train_mlp.py` | 4 | Trains the `Sequential` two-`Linear` MLP and the bare two-tensor MLP |
| `train_deep.py` | 5 | Trains `Linear → Tanh → Linear → Sigmoid` |
| `test_forge.py` | 6 | Eight unit tests, including two numeric gradient checks |
| `forge.ipynb` | all | Assembled notebook: three loss curves, gradient-check output, summary, holocron reading |
| `Reading_The_Holocron.txt` | 7 | The Part 7 reading (also the notebook's final cell) |

## How to run

Run everything from inside this `week13/` folder so the imports resolve:

```bash
cd week13
python tensor.py        # Part 1 acceptance checks: both print [2 2 2 2 2]
python train_mlp.py     # Part 4 loss curves
python train_deep.py    # Part 5 loss curve
python test_forge.py    # Part 6: PASS/FAIL per test
```

For the notebook, open `forge.ipynb` and use **Restart & Run All**. It needs `numpy` and `matplotlib`.

## Results (seed = 0)

| Run | Iter 0 | Iter 9 |
|---|---|---|
| Bare MLP (`alpha=0.1`) | 0.5813 | 0.0685 |
| `Sequential([Linear(2,3), Linear(3,1)])` (`alpha=0.01`) | 2.3343 | 0.0897 |
| `Linear → Tanh → Linear → Sigmoid` (`alpha=1`) | 1.0637 | 0.0470 |

All three match the handout and fall at every step. Both gradient checks agree with autograd to
about 1e-11, well inside the 1e-4 tolerance.

## Reproducibility

To match these numbers to the digit, each training run must call `np.random.seed(0)` first, then
create `data` and `target`, and only then create the weights or the model. Every `Tensor` draws its
id with `np.random.randint`, so creating tensors in a different order shifts every later random draw.
The curve then starts somewhere else, but its shape (falling at every step) stays the same.

## Gotchas

- `data` and `target` need `autograd=True`. Otherwise the loss is a plain tensor, `backward()` returns
  at once, and every gradient stays `None`.
- Only build tensors that lead to the loss. A side tensor such as `err = pred - target`, made just to
  look at it, leaves `pred` waiting for a child that never reports. To inspect a value, read `.data`.
- `'NoneType' object has no attribute 'data'` in `optim.step()` means a parameter never got a gradient:
  a backward branch on its path is missing.
