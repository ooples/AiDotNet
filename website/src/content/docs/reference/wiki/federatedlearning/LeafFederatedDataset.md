---
title: "LeafFederatedDataset<TInput, TOutput>"
description: "Represents a LEAF dataset with optional train/test splits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Represents a LEAF dataset with optional train/test splits.

## For Beginners

Many datasets ship with a training split (used to learn)
and a testing split (used to evaluate). This type groups those together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeafFederatedDataset(LeafFederatedSplit<,>,LeafFederatedSplit<,>)` | Initializes a new instance of the `LeafFederatedDataset` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Test` | Gets the optional test split. |
| `Train` | Gets the training split. |

