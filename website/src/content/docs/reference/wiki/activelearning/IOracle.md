---
title: "IOracle<TInput, TOutput>"
description: "Interface for oracles (labeling providers) in active learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for oracles (labeling providers) in active learning.

## For Beginners

An oracle is the source of labels for unlabeled data.
In real applications, this is typically a human expert. In experiments, it can be
a simulator using ground-truth labels.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLabelingCost()` | Gets the cost of labeling a sample. |
| `Label()` | Provides a label for a single sample. |
| `LabelBatch([])` | Provides labels for a batch of samples. |

