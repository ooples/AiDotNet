---
title: "ShuffleSplitter<T>"
description: "Monte Carlo cross-validation splitter that creates multiple random train/test splits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Basic`

Monte Carlo cross-validation splitter that creates multiple random train/test splits.

## For Beginners

Unlike K-Fold which systematically rotates through the data,
Monte Carlo CV (also called Shuffle-Split) randomly samples train/test sets multiple times.
Each split is independent - the same sample might be in the test set multiple times
across different splits.

## How It Works

**How It Works:**

**When to Use:**

- When you want flexibility in train/test proportions
- When data order doesn't matter
- As an alternative to K-Fold when you want more control

**Comparison to K-Fold:**

- K-Fold: Every sample appears in test exactly once
- Shuffle-Split: Samples may appear in test zero, one, or multiple times

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShuffleSplitter(Int32,Double,Nullable<Double>,Int32)` | Creates a new shuffle splitter (Monte Carlo CV). |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

