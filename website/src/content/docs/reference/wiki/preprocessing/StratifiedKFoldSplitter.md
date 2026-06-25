---
title: "StratifiedKFoldSplitter<T>"
description: "Stratified K-Fold cross-validation that preserves class distribution in each fold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Stratified K-Fold cross-validation that preserves class distribution in each fold.

## For Beginners

Stratified K-Fold is like regular K-Fold, but it ensures that
each fold has approximately the same proportion of each class as the original dataset.

## How It Works

**Why Stratification Matters:**
Imagine you have 90% cats and 10% dogs. With regular K-Fold, one fold might randomly
get no dogs at all! Stratification prevents this by ensuring each fold has ~90% cats and ~10% dogs.

**When to Use:**

- Classification problems (required for labels)
- Imbalanced datasets (more of one class than another)
- Any classification task - it's the industry standard!

**Industry Standard:** For classification tasks, ALWAYS prefer StratifiedKFold over regular KFold.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedKFoldSplitter(Int32,Boolean,Int32)` | Creates a new Stratified K-Fold cross-validation splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

