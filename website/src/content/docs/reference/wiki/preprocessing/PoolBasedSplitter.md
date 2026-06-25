---
title: "PoolBasedSplitter<T>"
description: "Pool-based active learning splitter that maintains labeled and unlabeled pools."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning`

Pool-based active learning splitter that maintains labeled and unlabeled pools.

## For Beginners

In pool-based active learning, we start with a small labeled dataset
and a large pool of unlabeled data. The model queries samples from the unlabeled pool
that it is most uncertain about, and an oracle (human expert) provides labels.

## How It Works

**How It Works:**

1. Start with a small initial labeled set (seed)
2. The rest becomes the unlabeled pool
3. Model queries uncertain samples from the pool
4. Oracle labels them, moving them to the labeled set

**When to Use:**

- Limited labeling budget
- Expensive manual annotation
- Interactive machine learning
- Iterative model improvement

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoolBasedSplitter(Double,Int32,PoolBasedSplitter<>.SelectionStrategy,Boolean,Int32)` | Creates a new pool-based active learning splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

