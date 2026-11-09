import numpy as np


def init_weights(vocab_size, embed_size=10, seed=1):
    """
    Initialize the four weight tensors using exact seeding and formula requirements:
      - embed: (V, embed_size) initialized with uniform random values in [-0.05, 0.05]
      - recurrent: (embed_size, embed_size) identity matrix
      - start: zeros of size (embed_size,)
      - decoder: (embed_size, V) initialized with uniform random values in [-0.05, 0.05]
    """
    np.random.seed(seed)
    
    embed = (np.random.rand(vocab_size, embed_size) - 0.5) * 0.1
    recurrent = np.eye(embed_size)
    start = np.zeros(embed_size)
    decoder = (np.random.rand(embed_size, vocab_size) - 0.5) * 0.1
    
    return embed, recurrent, start, decoder


def softmax(x):
    """
    Computes numerically stable softmax over a vector x by subtracting max(x).
    Returns a vector that sums to 1.0.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def predict(sent_indices, embed, recurrent, start, decoder):
    """
    Performs a forward pass given a list of word indices (sent_indices).
    
    Returns:
      - layers: List of layer dictionaries.
        * Layer 0: {'hidden': start}
        * Layer i (1 to len(sent_indices)): {'hidden': h_i, 'pred': output_probs}
      - total_loss: Sum of negative log-likelihood cross-entropy loss over all target tokens.
    """
    layers = [{'hidden': start.copy()}]
    total_loss = 0.0
    
    for idx in range(len(sent_indices)):
        prev_hidden = layers[-1]['hidden']
        current_word_idx = sent_indices[idx]
        
        # Guess the current word from the PREVIOUS hidden state, before the
        # network has seen it (otherwise it predicts a word it already read)
        logits = prev_hidden.dot(decoder)
        pred = softmax(logits)

        # target token for this layer prediction is the current word
        # sent_indices represents the sliced sentence input (e.g. tokens[0][1:])
        target_idx = sent_indices[idx]

        # Then take in the word: prev_hidden @ recurrent + embed[current_word_idx]
        hidden_state = prev_hidden.dot(recurrent) + embed[current_word_idx]
        
        # Accumulate cross-entropy loss: -log(P(target))
        # Clip pred to avoid log(0) numerical instability
        prob = np.clip(pred[target_idx], 1e-15, 1.0)
        total_loss += -np.log(prob)
        
        layers.append({
            'hidden': hidden_state,
            'pred': pred
        })
        
    return layers, total_loss


def compute_perplexity(total_loss, sequence_length):
    """
    Computes perplexity: exp(total_loss / sequence_length)
    """
    return np.exp(total_loss / sequence_length)


if __name__ == "__main__":
    # Self-contained validation check matching Part 2 requirement
    V = 82  # Vocabulary size for bAbI single-supporting-fact dataset
    embed, recurrent, start, decoder = init_weights(vocab_size=V, embed_size=10, seed=1)
    
    # Dummy sample sentence of length 4 (e.g., ['moved', 'to', 'the', 'bathroom.'])
    dummy_sent = [1, 2, 3, 4]
    
    layers, loss = predict(dummy_sent, embed, recurrent, start, decoder)
    ppl = compute_perplexity(loss, len(dummy_sent))
    
    print(f"Number of layers generated: {len(layers)} (Expected: {len(dummy_sent) + 1})")
    print(f"Untrained model loss: {loss:.4f}")
    print(f"Untrained perplexity: {ppl:.2f} (Expected: ~81.94)")
