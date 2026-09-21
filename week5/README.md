# Week 5: Trial of Reflection

We train a small network to predict a strike from three binary tells.
The target is an XOR pattern: foot or guard alone means strike, but both
or neither means hold. A single linear layer cannot fit this pattern.

## Files

- `part1_single_layer_fails.py`: train a single layer and observe failure.
- `part2_forward_hidden.py`: forward pass with a ReLU hidden layer.
- `part3_one_backprop_step.py`: one backpropagation update.
- `part4_full_training_loop.py`: full training and hidden-size sweep.
- `test_trial.py`: eight basic tests.
- `trial_of_reflection.ipynb`: parts assembled with short explanations.

## Run

Requires Python and NumPy. From the repository root:

```bash
python week5/test_trial.py
python week5/part4_full_training_loop.py
```

Open the notebook in Jupyter and choose Restart & Run All.
Its code imports the existing part files, so keep them in the same folder.

## Results

With alpha 0.2, 60 epochs, four hidden units, and seed 1, the final epoch
error is about 0.000015 and all four predictions are on the correct side
of 0.5. Hidden units 1 and 3 detect foot without guard and guard without
foot; units 0 and 2 are inactive on this dataset.

Using final epoch error below 0.01 as success, sizes 1 and 2 fail on all
three tested seeds, sizes 4 and 8 have mixed results, and size 16 succeeds
on all three. Size 4 is the smallest tested success on seed 1; size 16 is
the smallest tested size that meets that error threshold across all three seeds.
