## Diagram 1: Wider Network

```text
layer_0 --[weights_0_1 (3, 32), relu]--> layer_1 --[weights_1_2 (32, 1)]--> layer_2
(1, 3)                                   (1, 32)                            (1, 1)
```

**Total Weights:**  
$3 \times 32 + 32 \times 1 = 96 + 32 = 128$ total weights.

A wider hidden layer allows the network to learn many diverse feature combinations simultaneously, which can speed up training on complex pattern matching tasks. However, a tiny dataset having 128 weights like the Trial dataset (that has 4 samples) carries a high risk of overfitting or memorizing noise rather than generalizing well.


## Diagram 2: Deeper Network

```text
layer_0 --[weights_0_1 (3, 4), relu]--> layer_1 --[weights_1_2 (4, 4), relu]--> layer_2 --[weights_2_3 (4, 4), relu]--> layer_3 --[weights_3_4 (4, 1)]--> layer_4
(1, 3)                                  (1, 4)                                  (1, 4)                                     (1, 4)                            (1, 1)
```
**Total Weights:**  
 $3 \times 4 + 4 \times 4 + 4 \times 4 + 4 \times 1 = 12 + 16 + 16 + 4 = 48$

 A deeper network allows each next hidden layer to combine what was learned from the previous layer to model highly complex non-linear functions. However, deeper networks are much more sensitive to random weight initialization that could cause dead ReLU units or vanishing gradients.

## Diagram 3: Multi-output Network

```text
layer_0 --[weights_0_1 (3, 8), relu]--> layer_1 --[weights_1_2 (8, 4)]--> layer_2
(1, 3)                                  (1, 8)                             (1, 4)
```
**Total Weights:**  
$3 \times 8 + 8 \times 4 = 24 + 32 = 56$

Multi-output networks can perform multi label classification that leverages shared representations learned in the hidden layer across all outputs. However if the output tasks contradict each other, gradient updates from one output target might interfere with or degrade performance on another.


## Diagram 4: Mystery Network

Ten environmental features that are processed through three hidden layers to classify the target into one of four behavior categories.

```text
layer_0 --[weights_0_1 (10, 32), relu]--> layer_1 --[weights_1_2 (32, 32), relu]--> layer_2 --[weights_2_3 (32, 16), relu]--> layer_3 --[weights_3_4 (16, 4)]--> layer_4
(1, 10)                                  (1, 32)                                  (1, 32)                                  (1, 16)                             (1, 4)
```


**Total Weights:**  
$10 \times 32 + 32 \times 32 + 32 \times 16 + 16 \times 4 = 320 + 1024 + 512 + 64 = 1,920$

The architecture is wider at first but then narrows drastically allowing high dimensional feature extraction across 10 input signals before being compressed into 4 output classes. However, the almost 2 thousand weights across the four layer would require a larger dataset to avoid overfitting.

