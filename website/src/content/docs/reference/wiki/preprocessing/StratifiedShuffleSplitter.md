---
title: "StratifiedShuffleSplitter<T>"
description: "Stratified Monte Carlo cross-validation that preserves class distribution in random splits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Stratified Monte Carlo cross-validation that preserves class distribution in random splits.

## For Beginners

This combines:

- **Shuffle-Split:** Random train/test splits repeated multiple times
- **Stratification:** Each split preserves the original class distribution

## How It Works

**When to Use:**

- Classification problems with imbalanced classes
- When you want multiple random evaluations (like Monte Carlo CV)
- But also need to ensure each split has proper class representation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedShuffleSplitter(Int32,Double,Int32)` | Creates a new Stratified Shuffle splitter. |

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

