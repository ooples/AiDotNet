---
title: "JaccardIndex<T>"
description: "Jaccard Index for comparing clustering results against ground truth."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Jaccard Index for comparing clustering results against ground truth.

## For Beginners

Jaccard Index asks "How similar are the groupings?"

It compares every pair of points:

- "Are these two together in the true groups?"
- "Are these two together in the predicted groups?"

Then it calculates:

- Agreement = Both say together OR both say apart
- Jaccard = Pairs together in both / Pairs together in at least one

Values range from 0 (completely different) to 1 (identical clusterings).
Higher is better!

## How It Works

The Jaccard Index measures the similarity between two clusterings by computing
the ratio of pairs that are in the same cluster in both clusterings to pairs
that are in the same cluster in at least one clustering.

Jaccard = a / (a + b + c)
Where:

- a = pairs in same cluster in both clusterings
- b = pairs in same cluster only in true labels
- c = pairs in same cluster only in predicted labels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JaccardIndex` | Initializes a new JaccardIndex instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputePairConfusionMatrix(Vector<>,Vector<>)` | Computes the pair confusion matrix components. |

