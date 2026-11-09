# Week 12: Visions — Recurrent Neural Networks & BPTT

This repository contains the code, unit tests, data preprocessing scripts, and final Jupyter notebook for **COMP 3200 Deep Learning — Week 12 (Visions)**[cite: 1]. The objective of this assignment is to implement a Recurrent Neural Network (RNN) from scratch in NumPy and train it using Backpropagation Through Time (BPTT) for next-word prediction on sequential text[cite: 1, 4].

---

## 1. Directory & File Structure

```text
week12/
├── data/
│   └── qa1_train.txt          # bAbI single-supporting-fact training corpus (1,010 lines)
├── preprocess.py              # Loads 1,010 lines, builds sorted vocabulary & word2index
├── part2_The_Forward_Pass.py  # Weight initialization, stable softmax, & forward RNN pass
├── rnn_bptt.py                # Backward pass (BPTT) and exact weight tensor updates
├── train.py                   # 30,000-iteration training loop & perplexity logging
├── generate.py                # Next-word evaluation table on lines 1000–1009
├── test_visions.py            # Unit test suite (5 fast validation tests)
├── visions.ipynb              # Final assembled Jupyter notebook deliverable
├── perplexity_log.csv         # Saved training perplexity log (recorded every 1,000 iters)
├── weights.npz                # Saved trained model parameters (embed, recurrent, start, decoder)
└── README.md                  # Project documentation & execution guide
```[cite: 2]

> **Note on `.gitignore`**: The `data/` directory is excluded from version control to keep repository size lean[cite: 2, 22]. Each team member must download the raw corpus locally prior to running the pipeline[cite: 2, 22].

---

## 2. Dataset Setup

1. **Download**: Obtain the bAbI single-supporting-fact training file `qa1_single-supporting-fact_train.txt` from the course dataset repository[cite: 2, 22].
2. **Placement**: Save the downloaded file as `qa1_train.txt` inside `week12/data/`[cite: 2, 22].
3. **Line Partitioning**:
   - Lines `0`–`999` (first 1,000 lines): Used exclusively during the 30,000 training iterations[cite: 2, 5].
   - Lines `1000`–`1009` (last 10 lines): Out-of-training set reserved for the Part 5 next-word prediction demonstration[cite: 2, 6].

---

## 3. Key Architecture & Design Decisions

* **Deterministic Vocabulary**: Vocabulary construction uses `sorted(set(vocab))` across all 1,010 lines, ensuring an exact vocabulary size of $V=82$ and preserving numerical reproducibility across different execution environments[cite: 2, 3, 8].
* **Weight Initialization**:
  * `embed`: Shape $(82, 10)$, initialized uniformly in $[-0.05, 0.05]$[cite: 3].
  * `recurrent`: Shape $(10, 10)$, initialized to the **identity matrix** $I_{10}$ to prevent early signal erasure across timesteps[cite: 3, 9].
  * `start`: Shape $(10,)$, initialized to all zeros[cite: 3].
  * `decoder`: Shape $(10, 82)$, initialized uniformly in $[-0.05, 0.05]$[cite: 3].
* **BPTT Gradient Accumulation**: At intermediate hidden layers, the hidden delta $\delta_{\text{hidden}}^{(t)}$ sums both the local output prediction error and the backward-flowing error through recurrence[cite: 4]:
  $$\delta_{\text{hidden}}^{(t)} = \delta_{\text{output}}^{(t)} W_{\text{decoder}}^T + \delta_{\text{hidden}}^{(t+1)} W_{\text{recurrent}}^T$$[cite: 4]
* **Update Scaling**: All weight update deltas are scaled by $\frac{\alpha}{\text{len}(sent)}$ to ensure normalization across varying sentence lengths[cite: 4].

---

## 4. How to Run

Execute all command-line operations from inside the `week12/` directory[cite: 22].

### Step 1: Run Preprocessing Verification
Verify tokenization, vocabulary creation, and mapping:
```bash
python preprocess.py
```[cite: 2]

### Step 2: Run the Unit Test Suite
Confirm that all 5 required components (Preprocessing, Softmax, Forward shape, Loss reduction, and Hidden delta summation) pass:
```bash
python test_visions.py
```[cite: 2, 6]

### Step 3: Train the RNN Model
Train for 30,000 iterations over the first 1,000 lines ($\alpha = 0.001$). This exports `perplexity_log.csv` and `weights.npz`:
```bash
python train.py
```

### Step 4: Run Next-Word Evaluation
Evaluate next-word predictions on declarative test sentences past the training cut:
```bash
python generate.py
```

### Step 5: Notebook Execution
Launch Jupyter and open `visions.ipynb`[cite: 2]:
```bash
jupyter notebook visions.ipynb