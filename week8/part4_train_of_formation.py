import time
import numpy as np
from data import load_mnist, one_hot
from part1_memorize import relu, relu_d, evaluate


def train_batched(X_train, Y_train, X_val, Y_val, iterations=350, batch_size=100):
    """
    Trains a 784 -> 40 -> 10 network using mini-batches and dropout.
    
    Parameters:
        X_train, Y_train: Training data (1000 images, one-hot labels)
        X_val, Y_val: Validation data (1000 images, one-hot labels)
        iterations: Number of epochs (passes over the training set)
        batch_size: Number of images per mini-batch step
        
    Returns:
        W_0_1, W_1_2: Updated layer weights
        train_hist: List of training accuracy (running count) per iteration
        val_hist: List of validation accuracy per iteration
    """
    # Seed FIRST, then initialize weights
    np.random.seed(1)
    
    # Batch learning rate raised from 0.005 to 0.1
    alpha = 0.1
    hidden_size = 40
    
    # Initialize weights in [-0.1, 0.1)
    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1
    
    train_hist = []
    val_hist = []
    
    num_samples = X_train.shape[0]
    
    for epoch in range(iterations):
        correct_cnt = 0
        
        # Mini-batch inner loop
        for i in range(0, num_samples, batch_size):
            # Extract batch slices: shapes (100, 784) and (100, 10)
            batch_x = X_train[i : i + batch_size]
            batch_y = Y_train[i : i + batch_size]
            
            # Forward pass
            layer_0 = batch_x  # (100, 784)
            layer_1 = relu(np.dot(layer_0, W_0_1))  # (100, 40)
            
            # Apply Inverted Dropout (50% keep probability)
            mask = np.random.randint(2, size=layer_1.shape)  # (100, 40)
            layer_1 = layer_1 * mask * 2
            
            layer_2 = np.dot(layer_1, W_1_2)  # (100, 10)
            
            # Track running training accuracy (evaluated with dropout mask on)
            correct_cnt += np.sum(
                np.argmax(layer_2, axis=1) == np.argmax(batch_y, axis=1)
            )
            
            # Backward pass
            delta_2 = layer_2 - batch_y  # (100, 10)
            
            # Divide delta_2 by batch_size to average the gradient over the batch
            delta_2 = delta_2 / batch_size
            
            # Backpropagate through hidden layer and re-apply mask
            delta_1 = np.dot(delta_2, W_1_2.T) * relu_d(layer_1)  # (100, 40)
            delta_1 = delta_1 * mask  # Silenced units receive no gradient update
            
            # Weight updates
            W_1_2 -= alpha * np.dot(layer_1.T, delta_2)
            W_0_1 -= alpha * np.dot(layer_0.T, delta_1)
            
        # Record histories
        train_acc = correct_cnt / float(num_samples)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)
        
        train_hist.append(train_acc)
        val_hist.append(val_acc)
        
    return W_0_1, W_1_2, train_hist, val_hist


if __name__ == "__main__":
    # 1. Load data and carve 1,000-image subsets
    (X_train_full, y_train_full), _ = load_mnist()
    X_val = X_train_full[1000:2000]
    Y_val = one_hot(y_train_full[1000:2000])
    X_train = X_train_full[:1000]
    Y_train = one_hot(y_train_full[:1000])
    
    # Import single-image train function from Part 3 for timing comparison
    from part3_dropout import train_dropout

    print("Running Part 3 (One-at-a-time with dropout)...")
    t0 = time.perf_counter()
    W0_p3, W1_p3, tr_p3, val_p3 = train_dropout(
        X_train, Y_train, X_val, Y_val, iterations=350, dropout=True
    )
    t_single = time.perf_counter() - t0

    print("Running Part 4 (Batched with mini-batch size=100)...")
    t0 = time.perf_counter()
    W0_p4, W1_p4, tr_p4, val_p4 = train_batched(
        X_train, Y_train, X_val, Y_val, iterations=350, batch_size=100
    )
    t_batched = time.perf_counter() - t0

    speedup = t_single / t_batched

    # Report performance and benchmark results
    print("\n--- Benchmark Results ---")
    print(f"One-at-a-time execution time : {t_single:.4f} seconds")
    print(f"Mini-batched execution time   : {t_batched:.4f} seconds")
    print(f"Speed-up factor               : {speedup:.2f}x faster")
    print("--------------------------")
    print(f"Part 3 (Single-image) Final Val Acc : {val_p3[-1]:.4f}")
    print(f"Part 4 (Batched) Final Val Acc      : {val_p4[-1]:.4f}")
    
    # Confirm validation accuracy lands within ~2 percentage points
    acc_diff = abs(val_p3[-1] - val_p4[-1])
    print(f"Absolute Validation Difference     : {acc_diff:.4f}")
    