---
title: "DefaultCollateFunction<T>"
description: "Stacks equal-size tensors into a batch tensor along dimension 0."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Collation`

Stacks equal-size tensors into a batch tensor along dimension 0.

## How It Works

This is the default collation strategy, equivalent to PyTorch's default_collate.
All samples must have the same shape. The resulting batch tensor has shape
[N, ...sample_shape] where N is the number of samples.

## Methods

| Method | Summary |
|:-----|:--------|
| `Collate(IReadOnlyList<Tensor<>>)` |  |

