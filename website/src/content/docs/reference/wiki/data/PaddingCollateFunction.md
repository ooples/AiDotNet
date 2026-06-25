---
title: "PaddingCollateFunction<T>"
description: "Pads variable-length sequences to the maximum length in the batch, then stacks them."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Collation`

Pads variable-length sequences to the maximum length in the batch, then stacks them.

## How It Works

This collation strategy is essential for NLP and sequence models where inputs have
different lengths. Each sample is padded (or truncated) along the sequence dimension
to match the longest sample in the batch.

Assumes samples are 1D tensors (sequences of token IDs or values). The padding value
defaults to zero (common for PAD token in NLP).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaddingCollateFunction(,Nullable<Int32>)` | Creates a new padding collate function with an explicit pad value. |
| `PaddingCollateFunction(Nullable<Int32>)` | Creates a new padding collate function with a default pad value of zero. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Collate(IReadOnlyList<Tensor<>>)` |  |

