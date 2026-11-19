# Week 14: What to Remember
This week we showed, with a number, why a plain recurrent net’s gradient (the Week 12 RNN) vanishes or
explodes. We built an LSTM cell from operations the engine already differentiates, and say what each gate
does. We trained it with truncated BPTT and read its perplexity. Finally, we can say exactly what forcing f := 1 does to the cell state, and what it does not do to the gradient.

## Files

- `tensor.py`: Part 1: + index_select, cross_entropy, softmax
- `optim.py`: # (from Week 13, unchanged: SGD)
- `layers.py`: # Parts 2,4: + Embedding, CrossEntropyLoss, Linear(bias=); LSTMCell
- `exploding_gradients.py`: Part 3: vanishing / exploding / gated-vs-ungated
- `train_lstm.py`: Part 5: char-level LSTM training on the Old Tongue
- `test_week14` : Part 6: unit tests
- `what_to_remember.ipynb`: assembled notebook

## Just Run the Notebook

Open the notebook in Jupyter and choose Restart & Run All.
Its code imports the existing part files, so keep them in the same folder.