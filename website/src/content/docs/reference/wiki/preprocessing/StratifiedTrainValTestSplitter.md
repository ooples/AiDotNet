---
title: "StratifiedTrainValTestSplitter<T>"
description: "Stratified three-way split that preserves class distribution in all sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified`

Stratified three-way split that preserves class distribution in all sets.

## For Beginners

This creates train/validation/test splits while ensuring
each set has the same class proportions as the original data.

## How It Works

**Example:**
If original data has 60% Class A, 30% Class B, 10% Class C:

- Training set: ~60% A, ~30% B, ~10% C
- Validation set: ~60% A, ~30% B, ~10% C
- Test set: ~60% A, ~30% B, ~10% C

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedTrainValTestSplitter(Double,Double,Boolean,Int32)` | Creates a new stratified three-way splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |
| `SupportsValidation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

