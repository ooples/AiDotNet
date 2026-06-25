---
title: "RandIndex<T>"
description: "Rand Index and Adjusted Rand Index for comparing clustering results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Rand Index and Adjusted Rand Index for comparing clustering results.

## For Beginners

Rand Index asks "What fraction of pairs do we agree on?"

- If true labels say "together", do we also say "together"?
- If true labels say "apart", do we also say "apart"?

Adjusted Rand Index goes further by asking "How much better than random?"

- ARI = 0 means no better than random
- ARI = 1 means perfect agreement
- ARI can be negative if worse than random

## How It Works

The Rand Index measures the percentage of pair decisions that agree between
two clusterings. The Adjusted Rand Index corrects for chance agreement.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandIndex(Boolean)` | Initializes a new RandIndex instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |

