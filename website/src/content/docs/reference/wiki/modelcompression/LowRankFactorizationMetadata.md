---
title: "LowRankFactorizationMetadata<T>"
description: "Metadata for Low-Rank Factorization compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelCompression`

Metadata for Low-Rank Factorization compression.

## For Beginners

This metadata stores:

- Matrix dimensions (how the vector was reshaped)
- Rank used (how many singular values kept)
- Original length (for reconstruction)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LowRankFactorizationMetadata(Int32,Int32,Int32,Int32)` | Initializes a new instance of the LowRankFactorizationMetadata class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Cols` | Gets the number of columns in the reshaped matrix. |
| `OriginalLength` | Gets the original length of the weights array. |
| `Rank` | Gets the rank of the factorization. |
| `Rows` | Gets the number of rows in the reshaped matrix. |
| `Type` | Gets the compression type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetadataSize` | Gets the size in bytes of this metadata structure. |

