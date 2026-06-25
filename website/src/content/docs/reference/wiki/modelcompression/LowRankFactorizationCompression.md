---
title: "LowRankFactorizationCompression<T>"
description: "Implements Low-Rank Factorization compression using SVD-like decomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Implements Low-Rank Factorization compression using SVD-like decomposition.

## For Beginners

Low-Rank Factorization is like summarizing a book.

The concept:

- A weight matrix might be 1000×1000 = 1,000,000 parameters
- But the actual "information content" might be much smaller
- We can approximate it as: W ≈ A × B where A is 1000×50 and B is 50×1000
- Now we only store: 50,000 + 50,000 = 100,000 parameters (10x compression!)

How it works:

1. Treat the weight vector as a matrix (reshape it)
2. Perform approximate factorization (similar to SVD)
3. Keep only the top-k singular values/vectors
4. Store the factored matrices instead of the original

Benefits:

- Compression ratio is controlled by the rank k
- Works especially well for fully-connected layers
- Maintains smoothness in the weight space

Trade-offs:

- Need to choose the rank k (compression vs accuracy trade-off)
- Works best when weights have inherent low-rank structure

## How It Works

Low-Rank Factorization approximates weight matrices by decomposing them into products of
smaller matrices. This is based on the observation that many neural network weight matrices
are approximately low-rank, meaning they can be represented with fewer parameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LowRankFactorizationCompression(Int32,Double,Int32,Double)` | Initializes a new instance of the LowRankFactorizationCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApproximateSVD(Double[0:,0:],Int32,Int32)` | Performs approximate SVD using power iteration. |
| `Compress(Vector<>)` | Compresses weights using low-rank factorization. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses by reconstructing from U, S, V factors. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the compressed size. |
| `GetMatrixDimensions(Int32,Int32,Int32)` | Gets appropriate matrix dimensions for the weight vector. |
| `Normalize(Double[])` | Normalizes a vector and returns its norm. |

