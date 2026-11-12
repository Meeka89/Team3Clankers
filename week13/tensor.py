"""Parts 1-2: the autograd Tensor and its operations."""

# Part 1 TODO: implement Tensor with:
# - __init__(data, autograd=False, creators=None, creation_op=None, id=None)
# - children bookkeeping and all_children_accounted_for()
# - __add__, __repr__, and __str__
# - backward(grad=None, grad_origin=None), accumulating gradients and waiting
#   for all children before passing separate plain gradients to the creators

# Part 2 TODO: add forward and backward support for neg, sub, mul, mm,
# transpose, sum, and expand.

# Part 5 TODO: add tanh and sigmoid with their backward branches.
