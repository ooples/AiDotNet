---
title: "PackedSequenceBatch<T>"
description: "Represents a batch of packed (non-padded) variable-length sequences."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Collation`

Represents a batch of packed (non-padded) variable-length sequences.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PackedSequenceBatch(Tensor<>,Int32[],Int32[])` | Creates a new packed sequence batch. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Number of sequences in the batch. |
| `Data` | The flat tensor containing all sequence data concatenated. |
| `Lengths` | The length of each sequence in the batch (in sorted order). |
| `SortedIndices` | Maps sorted position back to original sample index. |

