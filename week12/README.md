# Week 12: Visions

Assignment 12 builds a recurrent neural network for next-word prediction.
Keep this week's Python files, tests, and assembled `visions.ipynb` notebook
in this folder.

## Dataset setup

Follow the [Week 12 download instructions](../DATASETS.md#everything-else-comes-from-one-public-repo)
and save the bAbI single-supporting-fact training file as `data/qa1_train.txt`
inside this folder. The repository already ignores `data/` directories.

Preprocessing reads the first 1,010 lines. Training uses only the first 1,000;
the last 10 are reserved for the next-word demonstration.

Run this week's scripts from the `week12/` directory.
