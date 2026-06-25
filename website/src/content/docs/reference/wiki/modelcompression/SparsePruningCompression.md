---
title: "SparsePruningCompression<T>"
description: "Implements sparse pruning compression by zeroing out small-magnitude weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Implements sparse pruning compression by zeroing out small-magnitude weights.

## For Beginners

Sparse pruning is like cleaning out your closet.

The idea:

- Many neural network weights are very small (close to zero)
- These tiny weights contribute little to the output
- We can set them to exactly zero without much accuracy loss
- Only store the non-zero weights and their positions

How it works:

1. Calculate a threshold (e.g., smallest 90% of weights by magnitude)
2. Set all weights below threshold to zero
3. Store only non-zero values with their indices

Benefits:

- Can achieve 90%+ sparsity (90% zeros) with minimal accuracy loss
- Sparse storage is very efficient (only store ~10% of weights)
- Works well combined with quantization or clustering

Example:

- Original: [0.001, 0.5, -0.002, 0.8, 0.003, -0.7]
- After 50% pruning: [0, 0.5, 0, 0.8, 0, -0.7]
- Sparse storage: values=[0.5, 0.8, -0.7], indices=[1, 3, 5]

## How It Works

Sparse pruning removes weights below a certain threshold, setting them to zero.
This creates sparsity in the model which can be exploited for efficient storage
using sparse matrix formats (only non-zero values and their indices are stored).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparsePruningCompression(Double,Double,Boolean)` | Initializes a new instance of the SparsePruningCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateThreshold(Vector<>)` | Calculates the pruning threshold based on configuration. |
| `Compress(Vector<>)` | Compresses weights by pruning small-magnitude values and storing in sparse format. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses sparse weights back to dense format. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the compressed size including sparse values and indices. |

