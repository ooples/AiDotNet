---
title: "GroupShuffleSplitter<T>"
description: "Random group-based train/test splits (Monte Carlo with groups)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

Random group-based train/test splits (Monte Carlo with groups).

## For Beginners

This is like ShuffleSplitter, but respects group boundaries.
Groups are randomly assigned to train or test, keeping all samples from the same
group together.

## How It Works

**When to Use:**

- When you want multiple random evaluations
- But need to keep groups together
- Good for getting variance estimates of group-based evaluation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupShuffleSplitter(Int32[],Int32,Double,Int32)` | Creates a new Group Shuffle splitter. |

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

