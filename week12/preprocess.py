"""Prepare the words and their numeric IDs for the Week 12 RNN."""

from pathlib import Path


data_path = Path(__file__).parent / "data" / "qa1_train.txt"

# Keep the first 1,010 lines in their original order.
tokens = []
with data_path.open(encoding="utf-8") as data:
    for _ in range(1010):
        line = data.readline()
        if not line:
            raise ValueError("qa1_train.txt must contain at least 1,010 lines.")

        # Split on spaces to preserve the tabs in the handout's example.
        words = line.lower().rstrip("\n").split(" ")
        tokens.append(words[1:])  # Remove the line number, keeping every word.

# A set keeps each word once. Sorting gives it a consistent position.
unique_words = set()
for sentence in tokens:
    for word in sentence:
        unique_words.add(word)

vocab = sorted(unique_words)

# Each word's ID is its position in vocab.
word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index


def words2indices(words):
    """Convert a list of words to their IDs without changing the order."""
    return [word2index[word] for word in words]


if __name__ == "__main__":
    print("Lines loaded:", len(tokens))
    print("Vocabulary size:", len(vocab))
    print("First three tokenized lines:")
    for sentence in tokens[:3]:
        print(sentence)
    print("First sentence as IDs:", words2indices(tokens[0]))
