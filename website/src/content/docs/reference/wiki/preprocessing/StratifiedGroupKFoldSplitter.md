---
title: "StratifiedGroupKFoldSplitter<T>"
description: "Stratified Group K-Fold cross-validation that keeps groups together while preserving class distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

Stratified Group K-Fold cross-validation that keeps groups together while preserving class distribution.

## For Beginners

This combines two important concepts:

1. Group K-Fold: Ensures all samples from the same group stay together
2. Stratification: Maintains the proportion of each class in train/test sets

## How It Works

**When to Use:**

- Medical studies where patients have multiple measurements AND classes are imbalanced
- Customer data where purchases from same customer should stay together AND you have rare categories
- Any grouped data with classification targets where class balance matters

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedGroupKFoldSplitter(Int32,Int32[],Boolean,Int32)` | Creates a new Stratified Group K-Fold splitter. |

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
| `WithGroups(Int32[])` | Sets the group assignments for samples. |

