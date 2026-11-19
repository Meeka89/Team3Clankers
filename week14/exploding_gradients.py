
# Carter Owens


import numpy as np

W = np.array([[1,4], [4, 1]])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



def relu(x):
    return np.maximum(0.0, x)


def vanishing():
    activation = sigmoid(np.array([1, 0.01]))
    acts = []

    for _ in range(10):
        activation = sigmoid(activation.dot(W))
        acts.append(activation)
   
    gradient = np.ones(2)
    gradients = []

    for act in reversed(acts):
        gradient = (gradient * act * (1 - act)).dot(W.T)
        gradients.append(gradient)

    return gradients


def exploding():
    activation = sigmoid(np.array([1, 0.01]))
    acts = []

    for _ in range(10):
        activation = relu(activation.dot(W))
        acts.append(activation)
   
    gradient = np.ones(2)
    gradients = []

    for act in reversed(acts):
        gradient = (gradient *(act > 0)).dot(W.T)
        gradients.append(gradient)
    
    return gradients




def lstm_cell_state_horizon(steps=50, forget_gate=True, seed=0):
    rng = np.random.RandomState(seed)
    c, norms = np.zeros(16), []
    for _ in range(steps):
        x = rng.randn(16)
        f_pre = rng.randn(16) # always drawn: same writes either way
        f = sigmoid(f_pre) if forget_gate else np.ones(16)
        i = sigmoid(rng.randn(16))
        c = f * c + i * np.tanh(x) # keep f of the old, write i of the new
        norms.append(float(np.max(np.abs(c))))
    return norms


if __name__ == "__main__":
    print("Vanishing:", [float(np.max(np.abs(g))) for g in vanishing()])
    print("Exploding:", [float(np.max(np.abs(g))) for g in exploding()])
    print("LSTM with forget gate:", lstm_cell_state_horizon(forget_gate=True)[-1])
    print("LSTM without forget gate:", lstm_cell_state_horizon(forget_gate=False)[-1])
