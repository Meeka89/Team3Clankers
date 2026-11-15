#---------optim.py----------#
#-------Andrew_Martin-------#


class SGD:
    #plain gradient descent: nudge every parameter against its gradient.

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters  # list of autograd Tensors to update
        self.alpha = alpha            # learning rate (step size)

    def zero(self):
        #Reset every gradient to zero without taking a step.
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        #Take one step downhill, then (by default) clear the gradients.
        for p in self.parameters:
            # Move against the gradient, scaled by the learning rate.
            p.data -= p.grad.data * self.alpha

            # backward() adds into p.grad with +=, so without this reset the
            # next step would also include this step's gradient.
            if zero:
                p.grad.data *= 0
