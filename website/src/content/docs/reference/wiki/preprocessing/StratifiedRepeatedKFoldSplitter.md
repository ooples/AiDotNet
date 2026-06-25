---
title: "StratifiedRepeatedKFoldSplitter<T>"
description: "Stratified Repeated K-Fold cross-validation combining stratification with multiple repeats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Stratified Repeated K-Fold cross-validation combining stratification with multiple repeats.

## For Beginners

This combines the benefits of:

- **Stratification:** Preserves class distribution in each fold
- **Repetition:** Runs multiple times for more stable estimates

## How It Works

**When to Use:**

- Classification problems with need for very reliable estimates
- Imbalanced datasets where stratification is important
- Model comparison where small differences matter

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedRepeatedKFoldSplitter(Int32,Int32,Int32)` | Creates a new Stratified Repeated K-Fold splitter. |

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

