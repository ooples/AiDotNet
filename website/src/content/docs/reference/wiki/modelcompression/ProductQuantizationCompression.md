---
title: "ProductQuantizationCompression<T>"
description: "Implements Product Quantization (PQ) compression for model weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Implements Product Quantization (PQ) compression for model weights.

## For Beginners

Product Quantization is like organizing a closet using multiple small bins.

Instead of trying to compress all your clothes in one big box:

1. Divide clothes into categories (shirts, pants, socks)
2. For each category, pick a few representative items
3. Store only which representative each item is most similar to

For neural network weights:

- Divide each weight vector into M smaller pieces (subvectors)
- For each piece, find K cluster centers (codebook)
- Replace each subvector with its nearest codebook entry

Benefits:

- Better accuracy than global clustering for the same compression ratio
- Very efficient for high-dimensional weight vectors
- Commonly used in production systems (e.g., FAISS library)

Example:

- 1024-dimensional weight vector divided into 8 subvectors of 128 dimensions each
- Each subvector has 256 possible codes (8-bit quantization)
- Original: 1024 × 32 bits = 32,768 bits
- Compressed: 8 × 8 bits + codebook = ~64 bits + codebook
- Massive compression with minimal accuracy loss!

## How It Works

Product Quantization is a powerful compression technique that divides weight vectors into subvectors
and quantizes each subvector separately using its own codebook. This provides a good balance between
compression ratio and reconstruction accuracy.

**Important Limitation:** This implementation is designed for compressing a single weight vector.
Traditional PQ achieves compression by training codebooks on multiple vectors and amortizing codebook storage.
For single-vector compression, the codebook overhead may exceed the original data size.

**When to use this compressor:**

- When you have very high-dimensional weight vectors (thousands of dimensions)
- When reconstruction quality is more important than compression ratio
- When you plan to extend to batch compression of multiple similar vectors

**For better single-vector compression:**

- Consider `WeightClusteringCompression` for simpler k-means clustering
- Consider `HuffmanEncodingCompression` for lossless entropy coding
- Consider `DeepCompression` for a multi-stage pipeline

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProductQuantizationCompression(Int32,Int32,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the ProductQuantizationCompression class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(Vector<>)` | Compresses weights using Product Quantization. |
| `CreateCodebookForSubvector([])` | Creates a codebook for a single subvector using K-means clustering. |
| `Decompress(Vector<>,ICompressionMetadata<>)` | Decompresses weights by reconstructing from codebooks and codes. |
| `GetCompressedSize(Vector<>,ICompressionMetadata<>)` | Gets the compressed size including codebooks and codes. |

