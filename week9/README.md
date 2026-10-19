# Week 8: Forms

This week we demonstrate why a network has to be more than linear. We illustrate the need for nonlinearity, and how linear layers collapse into a single layer. We implement the different forms currently at our disposal: sigmoid, tanh, relu, and softmax. We show its use on the MNIST data and compare results of the different forms on the hidden layer.

## Files

- `data.py`: loads in MNIST data
- `forms.py`: the activation functions and respective derivatives; generate plots
- `part2_nonlinearity.py`: illustrate how two linear layers collapse to one; shows the need for nonlinearity
- `part3_upgrade.py`: tanh for hidden leayer and softmax output on the MNIST data
- `part4_compare.py`: compares relu, tanh, and sigmoid and their effects on a hidden layer
- `forms.ipynb`: assembled notebook

## Just Run the Notebook

Open the notebook in Jupyter and choose Restart & Run All.
Its code imports the existing part files, so keep them in the same folder.