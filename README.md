# Self-Pruning Neural Network

## Overview
This project implements a **self-pruning neural network** where the model learns to remove unnecessary weights during training.

Instead of pruning after training, each weight is associated with a learnable **gate parameter** that controls whether the connection is active or pruned.

---

## Core Idea

Each weight has a gate:

gate = sigmoid(score)

- If gate ≈ 1 → weight is active  
- If gate ≈ 0 → weight is pruned  

The model learns these gates automatically using a sparsity penalty.

---

## Methodology

### 1. Custom Layer: PrunableLinear
- Contains:
  - weights
  - bias
  - gate_scores
- Forward pass:
  - gates = sigmoid(gate_scores)
  - pruned_weights = weights × gates

---

### 2. Loss Function

Total Loss = CrossEntropyLoss + λ × SparsityLoss

- CrossEntropyLoss → classification accuracy  
- SparsityLoss → encourages pruning  

SparsityLoss is computed as:

SparsityLoss = sum of all gate values (L1 norm)

---

### 3. Training Setup

- Dataset: CIFAR-10  
- Training samples: 10,000 (subset)  
- Epochs: 5  
- Optimizer: Adam  
- Separate learning rates:
  - weights: 1e-3  
  - gates: 0.05  

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 1e-5   | ~50%     | Moderate |
| 1e-4   | ~30–40%  | High     |
| 1e-3   | ~10%     | Very High |

---

## Observations

- Increasing λ increases sparsity  
- Very high λ leads to over-pruning and accuracy drop  
- Moderate λ gives best trade-off  

---

## Gate Distribution

Plots show:
- Large spike near 0 → pruned weights  
- Remaining values → important connections  

---

## Files Included

- `main.py` → complete implementation  
- `report.md` → detailed results and explanation  
- `gates_lambda_*.png` → gate distribution plots  

---

## How to Run

```bash
python main.py
