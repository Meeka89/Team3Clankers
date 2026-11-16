


from tensor import Tensor
from layers import Linear, Sequential, Layer
from optim import SGD
import numpy as np

#   --- Not Bare MLP ---
np.random.seed(0)
data = Tensor([[0,0],[0,1],[1,0],[1,1]], autograd=True)
target = Tensor([[0],[1],[0],[1]], autograd=True)
model = Sequential([Linear(2,3), Linear(3,1)])
optim = SGD(parameters=model.get_parameters(), alpha=0.01)

print("--- two-Linear model (alpha=0.1) ---")
for i in range(10):
    pred = model.forward(data)
    loss = ((pred - target) * (pred - target)).sum(0)
    loss.backward()
    optim.step()
    print(f" iter {i}: loss {float(loss.data[0]):.6f}")


#      --- Bare MLP ---
w = [Tensor(np.random.rand(2, 3), autograd=True),
    Tensor(np.random.rand(3, 1), autograd=True)]
print("--- bare MLP (alpha=0.1) ---")
for i in range(10):
    pred = data.mm(w[0]).mm(w[1])
    loss = ((pred - target) * (pred - target)).sum(0)
    loss.backward(Tensor(np.ones_like(loss.data)))
    for w_ in w:
        w_.data -= w_.grad.data * 0.1
        w_.grad.data *= 0
    print(f" iter {i}: loss {float(loss.data[0]):.6f}")
