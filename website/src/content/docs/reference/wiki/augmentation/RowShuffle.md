---
title: "RowShuffle<T>"
description: "Shuffles rows within a batch of tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Shuffles rows within a batch of tabular data.

## For Beginners

Row shuffling randomly reorders the samples in your data.
While this doesn't create new data, it ensures the model doesn't learn from the
order of samples, which is especially important when data has natural ordering.

## How It Works

**When to use:**

- When data has natural temporal or sequential ordering
- As part of mini-batch training to randomize batch composition
- When consecutive samples might be correlated

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RowShuffle(Double)` | Creates a new row shuffle augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ShuffleWithLabels(Matrix<>,Vector<>,AugmentationContext<>)` | Shuffles data and labels together, maintaining row correspondence. |

