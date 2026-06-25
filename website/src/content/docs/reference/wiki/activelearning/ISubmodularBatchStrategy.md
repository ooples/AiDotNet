---
title: "ISubmodularBatchStrategy<T, TInput, TOutput>"
description: "Interface for submodular batch selection strategies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for submodular batch selection strategies.

## For Beginners

Submodular functions have a "diminishing returns" property:
adding more similar samples provides less benefit. This naturally encourages diversity.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMarginalGain(IReadOnlyList<Int32>,Int32,IDataset<,,>)` | Computes the marginal gain of adding a sample to the current selection. |
| `GreedyMaximization(Int32[],IDataset<,,>,Int32)` | Performs greedy submodular maximization to select a batch. |

