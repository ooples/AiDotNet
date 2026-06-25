---
title: "StratifiedKFoldStrategy<T>"
description: "Stratified K-Fold: K-Fold that preserves the percentage of samples for each class."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Stratified K-Fold: K-Fold that preserves the percentage of samples for each class.

## For Beginners

Stratified K-Fold is essential for classification problems, especially
with imbalanced classes:

- Each fold has approximately the same class distribution as the full dataset
- Prevents folds where a rare class is entirely missing
- Produces more reliable estimates for imbalanced datasets

## How It Works

**Example:** If your data has 70% class A and 30% class B, each fold will have
approximately 70% A and 30% B (not 100% A or 100% B which could happen with regular K-Fold).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedKFoldStrategy(Int32,Boolean,Nullable<Int32>)` | Initializes Stratified K-Fold cross-validation. |

