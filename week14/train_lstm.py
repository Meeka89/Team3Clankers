import numpy as np
import matplotlib.pyplot as plt

from tensor import Tensor
from layers import Embedding, LSTMCell, CrossEntropyLoss
from optim import SGD


np.random.seed(0)
embedding_size = 512
hidden_size = 512
batch_size = 16
bptt = 25
alpha = 0.032

text = open("data/qa_oldtongue.txt", encoding="utf-8").read()
vocab = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
data = np.array([char_to_idx[ch] for ch in text])

# Split the text into 16 continuous sequences
data = data[:len(data) // batch_size * batch_size]
data = data.reshape(batch_size, -1)

embedding = Embedding(len(vocab), embedding_size)
model = LSTMCell(embedding_size, hidden_size, len(vocab))
model.w_ho.weight.data *= 0
loss_fn = CrossEntropyLoss()

optimizer = SGD(
    embedding.get_parameters() + model.get_parameters(),
    alpha=alpha
)

hidden = model.init_hidden(batch_size)
total_loss = 0
total_steps = 0
perplexities = []

batch_num = 0
for start in range(0, data.shape[1] - 1, bptt):
    if batch_num == 60:
        break

    end = min(start + bptt, data.shape[1] - 1)

    hidden = (
        Tensor(hidden[0].data, autograd=True),
        Tensor(hidden[1].data, autograd=True)
    )

    chunk_loss = None

    for t in range(start, end):
        x = Tensor(data[:, t])
        target = Tensor(data[:, t + 1])

        x = embedding.forward(x)
        output, hidden = model.forward(x, hidden)
        loss = loss_fn.forward(output, target)

        chunk_loss = loss if chunk_loss is None else chunk_loss + loss

    chunk_loss.backward()
    optimizer.step()

    total_loss += float(chunk_loss.data)
    total_steps += end - start
    perplexity = np.exp(total_loss / total_steps)
    perplexities.append(perplexity)

    batch_num += 1
    print(f"Batch {batch_num}: perplexity = {perplexity:.2f}")

# Save the training curve for the final notebook
plt.plot(range(1, len(perplexities) + 1), perplexities)
plt.xlabel("Batch")
plt.ylabel("Running perplexity")
plt.title("LSTM Training")
plt.savefig("training_curve.png")
plt.show()
