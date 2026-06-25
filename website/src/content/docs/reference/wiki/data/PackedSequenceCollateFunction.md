---
title: "PackedSequenceCollateFunction<T>"
description: "Packs variable-length sequences into a contiguous tensor without padding, along with sequence lengths for reconstruction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Collation`

Packs variable-length sequences into a contiguous tensor without padding, along with
sequence lengths for reconstruction.

## How It Works

Packed sequences are more memory-efficient than padded batches because they don't store
padding tokens. The output contains a flat data tensor of all sequences concatenated,
plus a lengths tensor indicating how many elements belong to each sequence.
This is equivalent to PyTorch's pack_padded_sequence.

Sequences are sorted by length (descending) for efficient RNN processing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PackedSequenceCollateFunction(Boolean)` | Creates a new packed sequence collate function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Collate(IReadOnlyList<Tensor<>>)` |  |

