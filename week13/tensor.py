"""Part 1: the autograd Tensor."""

# Part 1 TODO: implement Tensor with:
# - __init__(data, autograd=False, creators=None, creation_op=None, id=None)
# - children bookkeeping and all_children_accounted_for()
# - __add__, __repr__, and __str__
# - backward(grad=None, grad_origin=None), accumulating gradients and waiting
#   for all children before passing separate plain gradients to the creators
