---
title: "KFoldSplitter<T>"
description: "K-Fold cross-validation splitter that divides data into k equal folds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

K-Fold cross-validation splitter that divides data into k equal folds.

## For Beginners

K-Fold cross-validation is one of the most important techniques
in machine learning for getting reliable performance estimates.

## How It Works

**How It Works:**

1. Divide data into k equal parts (folds)
2. For each fold:
- Use that fold as the test set
- Use the remaining k-1 folds as training
3. Average the results across all k evaluations

**Visual Example (5-Fold):**

**Industry Standard:**

- k=5 for large datasets (>10,000 samples)
- k=10 for smaller datasets (1,000-10,000 samples)

**Why Use K-Fold?**

- More reliable than a single train/test split
- Every sample gets tested exactly once
- Good for limited data where you can't afford to hold out a large test set

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KFoldSplitter(Int32,Boolean,Int32)` | Creates a new K-Fold cross-validation splitter. |

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

