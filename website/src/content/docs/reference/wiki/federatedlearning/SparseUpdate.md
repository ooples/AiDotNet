---
title: "SparseUpdate<T>"
description: "Represents a sparse update: only a subset of indices have non-zero values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Represents a sparse update: only a subset of indices have non-zero values.
This is the communication-efficient representation used by SparseLoRA.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseUpdate(Int32[],[],Int32)` | Creates a new sparse update. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Density` | Effective sparsity: fraction of elements that are non-zero. |
| `Indices` | Indices of non-zero elements. |
| `NnzCount` | Number of non-zero elements. |
| `TotalLength` | Total length of the full dense vector. |
| `Values` | Values at the non-zero indices. |

