# Analyzing Latent Representations in Poker Flops Using Autoencoders

## Overview
This project analyzes **poker flops** using **neural networks**, specifically **autoencoders**, to learn meaningful representations of flop characteristics. The goal is to capture patterns and structures within all possible flops using various encoded attributes and statistical properties.

## Data Generation
The dataset consists of **all 22,100 possible flops** from a standard 52-card deck. Each card is represented using **one-hot encoding** for:
- **Rank**: 13 values (2–9, T, J, Q, K, A)
- **Suit**: 4 values (Spades, Hearts, Clubs, Diamonds)

These encodings are concatenated to create a **"flop vector"**, resulting in a **51-dimensional vector** per flop. 

### **Additional Flop Attributes**
Other important properties extracted from each flop include:
- **Suitedness**
- **Pairness**
- **Connectedness**
- **High-Low Texture**
- **High Card**
- **Straightness**

The criteria and index mappings for these attributes are available in **[definitions.txt](definitions.txt)**, and the functions for computation are in **[utils.py](utils.py)**.

## Example Encoding
### **Flop: ['5s', '7h', 'Ts']**
The one-hot encoded vector representation:
```
[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  1., 0., 0., 0.]+
[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,  0., 1., 0., 0.]+
[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,  1., 0., 0., 0.]
```
This representation enables the neural network to learn structured relationships between flop configurations.
To see examples of all the flop attributes refer to [sample_data](sample_data.ipynb).

## Usage
1. Generate all possible flops and encode them.
2. Train the autoencoder on the encoded flop dataset.
3. Evaluate the latent space representation.

For more details, check the **README.md** and related files.

## Future Work
- Implement clustering techniques to group similar flops.
- Extend analysis to **turn** and **river** scenarios.
- Integrate with **poker decision-making models**.
- Experiment with **different neural network architectures**.

---
