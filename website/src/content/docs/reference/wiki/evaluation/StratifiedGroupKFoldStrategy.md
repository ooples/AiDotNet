---
title: "StratifiedGroupKFoldStrategy<T>"
description: "Stratified Group K-Fold: combines stratification and group constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Stratified Group K-Fold: combines stratification and group constraints.

## For Beginners

This strategy ensures both:

- Class distribution is preserved in each fold (stratification)
- Groups are kept together (no leakage between train/validation)

## How It Works

**Example use case:** Medical study where:

- Multiple samples from the same patient (group = patient ID)
- Classes are imbalanced (disease vs. healthy)
- Need both group integrity and class balance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedGroupKFoldStrategy(Int32,Int32[],Nullable<Int32>)` | Initializes Stratified Group K-Fold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SetGroups(Int32[])` | Sets the group identifiers. |

