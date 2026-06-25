---
title: "AntiClusteringSplitter<T>"
description: "Anti-clustering splitter that maximizes diversity within each split."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Anti-clustering splitter that maximizes diversity within each split.

## For Beginners

While clustering groups similar items together,
anti-clustering does the opposite - it ensures each group (train/test)
contains a diverse mix of samples covering the entire data space.

## How It Works

**How It Works:**

1. Compute pairwise distances between all samples
2. Iteratively assign samples to train/test sets
3. At each step, maximize the diversity within each set

**When to Use:**

- When you want both train and test to be representative
- Survey sampling where you want diverse groups
- A/B testing to ensure comparable groups

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AntiClusteringSplitter(Double,Boolean,Int32)` | Creates a new anti-clustering splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

