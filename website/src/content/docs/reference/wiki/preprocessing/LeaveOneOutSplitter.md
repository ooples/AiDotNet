---
title: "LeaveOneOutSplitter<T>"
description: "Leave-One-Out cross-validation where each sample is the test set once."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Leave-One-Out cross-validation where each sample is the test set once.

## For Beginners

Leave-One-Out (LOO) is the extreme version of K-Fold where k = n (number of samples).
Each sample is tested individually while all other samples are used for training.

## How It Works

**How It Works:**

**Pros:**

- Uses maximum training data (n-1 samples)
- Every sample gets tested
- No randomness - results are deterministic

**Cons:**

- Very slow: requires n model fits
- High variance in estimates
- Only practical for small datasets (<100 samples)

**When to Use:**

- Very small datasets where every sample matters
- When you need deterministic (non-random) evaluation
- Medical/scientific studies with limited data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeaveOneOutSplitter` | Creates a new Leave-One-Out cross-validation splitter. |

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

