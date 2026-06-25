---
title: "IBatchStrategy<T, TInput, TOutput>"
description: "Interface for batch selection strategies in active learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for batch selection strategies in active learning.

## For Beginners

When selecting multiple samples for labeling at once (batch mode),
we need to ensure diversity among the selected samples. Simply taking the top-N by score
might select very similar samples, which wastes labeling budget.

## How It Works

**Batch Selection Approaches:**

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityTradeoff` | Gets or sets the trade-off parameter between informativeness and diversity. |
| `Name` | Gets the name of the batch selection strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiversity(,)` | Computes pairwise diversity between samples. |
| `SelectBatch(Int32[],Vector<>,IDataset<,,>,Int32)` | Selects a diverse batch of samples from candidates. |

