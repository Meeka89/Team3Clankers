#----------preprocess.py----------#
#-------Andrew_Martin-------------#


import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(limit=None): # change this to 50 for fast tests
    """Read the corpus and return everything the later parts need."""
    # Read the files, one review / label per line
    with open(os.path.join(DATA_DIR, "reviews.txt"), encoding="utf-8") as f:
        reviews = [line.rstrip("\n") for line in f][:limit]
    with open(os.path.join(DATA_DIR, "labels.txt"), encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f][:limit]

    # Split each review into words, then number every distinct word
    tokens = [review.split() for review in reviews]
    vocab = sorted(set(word for review in tokens for word in review))
    word2index = {word: i for i, word in enumerate(vocab)}

    # Turn each review into word numbers: in order, and as a set
    ordered = [[word2index[word] for word in review] for review in tokens]
    as_sets = [list(set(review)) for review in ordered]

    # positive -> 1, negative -> 0
    targets = [1 if label == "positive" else 0 for label in labels]

    return vocab, word2index, as_sets, ordered, targets


if __name__ == "__main__":
    vocab, word2index, as_sets, ordered, targets = load()
    print("Vocabulary size:", len(vocab))
    print("Reviews:", len(targets))
    print("Train / test:", len(targets[:24000]), "/", len(targets[24000:]))