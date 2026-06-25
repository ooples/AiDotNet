---
title: "BootstrapSplitter<T>"
description: "Bootstrap sampling with out-of-bag (OOB) samples as the test set."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap`

Bootstrap sampling with out-of-bag (OOB) samples as the test set.

## For Beginners

Bootstrap is a resampling technique that creates training sets
by randomly sampling WITH replacement from your data.

## How It Works

**How It Works:**

1. Randomly select n samples from n total (with replacement)
2. Some samples will be picked multiple times, others not at all
3. The samples NOT picked (~36.8% on average) form the "out-of-bag" (OOB) test set

**Key Property:**
Each sample has a ~63.2% chance of being in the training set and ~36.8% chance
of being in the OOB test set. This is because P(not selected) = (1-1/n)^n ≈ 1/e.

**When to Use:**

- Error estimation with variance estimates
- When you want to use all data for training
- Foundation for bagging and random forests

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapSplitter(Int32,Int32)` | Creates a new bootstrap splitter. |

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

