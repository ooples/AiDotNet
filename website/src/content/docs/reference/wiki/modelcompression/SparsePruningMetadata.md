---
title: "SparsePruningMetadata<T>"
description: "Metadata for sparse pruning compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for sparse pruning compression.

## For Beginners

This metadata stores:

- Indices of non-zero values (so we know where to put them during decompression)
- Original vector length (to reconstruct the right size)
- Threshold used (for reference)
- Actual sparsity achieved (for statistics)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparsePruningMetadata(Int32[],Int32,Double,Double)` | Initializes a new instance of the SparsePruningMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualSparsity` | Gets the actual sparsity achieved (fraction of zeros). |
| `NonZeroIndices` | Gets the indices of non-zero values. |
| `OriginalLength` | Gets the original length of the weights array. |
| `Threshold` | Gets the threshold used for pruning. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

