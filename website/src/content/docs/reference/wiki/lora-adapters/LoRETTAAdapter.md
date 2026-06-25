---
title: "LoRETTAAdapter"
description: "LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRETTA (Low-Rank Economic Tensor-Train Adaptation) adapter for parameter-efficient fine-tuning.

## For Beginners

LoRETTA is an advanced version of LoRA that uses "tensor-train decomposition"!

Standard LoRA uses two matrices (A and B) to approximate weight changes:

- Matrix A: Compresses input to rank dimensions
- Matrix B: Expands back to output dimensions
- Parameters: inputSize × rank + rank × outputSize

LoRETTA uses multiple small "cores" chained together:

- Instead of 2 large matrices, use many small tensors
- Each core captures local correlations
- The cores are "contracted" (multiplied in sequence)
- Can express more complex patterns with fewer parameters

Why tensor-train decomposition?

1. More expressive: Can capture higher-order correlations
2. More efficient: Fewer parameters than matrix factorization
3. Better compression: Exploits structure in weight updates
4. Scalable: Grows logarithmically with dimensions

Example parameter counts for 1000×1000 layer:

- Full update: 1,000,000 parameters
- Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
- LoRETTA (rank=4, 3 cores): ~6,000 parameters (99.4% reduction, even better!)

Key parameters:

- ttRank: Controls compression (like LoRA's rank but more powerful)
- numCores: How many tensor cores in the chain (typically 3-5)
- alpha: Scaling factor for the adaptation strength

When to use LoRETTA:

- Maximum parameter efficiency needed
- Weight updates have higher-order structure
- You have very large layers to adapt
- Standard LoRA isn't expressive enough at low ranks

Reference:
Tensor-train decomposition: I. V. Oseledets, "Tensor-train decomposition,"
SIAM J. Scientific Computing, 2011.

## How It Works

LoRETTA extends LoRA by using tensor-train decomposition instead of simple matrix factorization.
Instead of representing weight updates as W = A × B, LoRETTA uses a tensor-train decomposition
that captures higher-order correlations with even fewer parameters.

Tensor-train decomposition represents a high-dimensional tensor as a sequence of lower-dimensional
"cores" that are contracted together. For a weight matrix W of size (m × n), the tensor-train
representation is:

W[i,j] = G1[i] × G2 × G3 × ... × Gd[j]

where each core Gk has dimensions (r_{k-1} × n_k × r_k), and r_k are the TT-ranks.
The boundary ranks are r_0 = r_d = 1.

