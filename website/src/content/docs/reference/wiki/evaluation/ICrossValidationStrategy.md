---
title: "ICrossValidationStrategy<T>"
description: "Defines a cross-validation splitting strategy."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Evaluation.CrossValidation`

Defines a cross-validation splitting strategy.

## For Beginners

Cross-validation strategies determine how to split your data into
training and validation sets. Different strategies are appropriate for different data types:

- **K-Fold:** Standard approach for most data
- **Stratified K-Fold:** Preserves class distribution (for classification)
- **Time Series Split:** Respects temporal order (for time series)
- **Leave-One-Out:** Maximum data usage but computationally expensive

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of this strategy suitable for documentation. |
| `Name` | Gets the name of this cross-validation strategy. |
| `NumSplits` | Gets the number of splits (folds) this strategy will generate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Int32,ReadOnlySpan<>)` | Generates train/validation index splits for the given data size. |

