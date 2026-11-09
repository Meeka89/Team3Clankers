#----------rnn_bptt.py----------#
#---------Andrew_Martin---------#

import numpy as np

from part2_The_Forward_Pass import predict


def backward(layers, sent, recurrent, decoder):
    """Fill in 'output_delta' and 'hidden_delta' on every layer, last to first."""
    one_hot = np.eye(decoder.shape[1])

    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]

        # Layer 0 is the start state
        # the delta pulled back from layer 1 through the recurrence.
        if layer_idx == 0:
            layer['hidden_delta'] = layers[1]['hidden_delta'].dot(recurrent.T)
            continue

        # Error on this step's guess
        target = sent[layer_idx - 1]
        layer['output_delta'] = layer['pred'] - one_hot[target]

        # Piece 1: error coming down from this step's own prediction
        new_hidden_delta = layer['output_delta'].dot(decoder.T)

        if layer_idx == len(layers) - 1:
            # Last layer has no future step, so piece 1 is all there is
            layer['hidden_delta'] = new_hidden_delta
        else:
            # Piece 2: error coming back from the next step through recurrent
            layer['hidden_delta'] = (new_hidden_delta
                + layers[layer_idx + 1]['hidden_delta'].dot(recurrent.T))


def update(layers, sent, embed, recurrent, start, decoder, alpha):
    """Exact update rule scaled by alpha / len(sent)."""
    step = alpha / len(sent)

    # The gradient for each state is stored one layer later
    for layer_idx, layer in enumerate(layers[1:]):
        # Decoder: the state one layer back made this layer's guess.
        decoder -= np.outer(layers[layer_idx]['hidden'],
                            layer['output_delta']) * step

        # The last state makes no guess, so the last word gets no update.
        if layer_idx + 2 < len(layers):
            grad_h = layers[layer_idx + 2]['hidden_delta']
            embed[sent[layer_idx]] -= grad_h * step  
            recurrent -= np.outer(layers[layer_idx]['hidden'], grad_h) * step


def train_step(sent, embed, recurrent, start, decoder, alpha=0.001):
    """Forward + backward + update on one sentence. Returns its perplexity."""
    layers, loss = predict(sent, embed, recurrent, start, decoder)
    backward(layers, sent, recurrent, decoder)
    update(layers, sent, embed, recurrent, start, decoder, alpha)
    return np.exp(loss / len(sent))  # per-word mean, not the total


if __name__ == "__main__":
    from preprocess import tokens, vocab, words2indices
    from part2_The_Forward_Pass import init_weights

    embed, recurrent, start, decoder = init_weights(len(vocab))
    sent = words2indices(tokens[0][1:])

    # Quick check: perplexity on one fixed sentence should drop.
    first = train_step(sent, embed, recurrent, start, decoder)
    for _ in range(99):
        last = train_step(sent, embed, recurrent, start, decoder)
    print(f"Perplexity on tokens[0][1:]: {first:.2f} -> {last:.2f} after 100 steps")
