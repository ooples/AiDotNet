---
title: "DoubleCVSplitter<T>"
description: "Double Cross-Validation splitter (alias for Nested CV with equal inner/outer folds)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Nested`

Double Cross-Validation splitter (alias for Nested CV with equal inner/outer folds).

## For Beginners

Double CV is a common configuration of Nested CV where both
the inner and outer loops use the same number of folds (typically 5 or 10).

## How It Works

**When to Use:**

- When you need unbiased performance estimation during hyperparameter search
- For comparing different model families fairly
- When computational resources allow (k² model fits)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DoubleCVSplitter(Int32,Boolean,Int32)` | Creates a new Double CV splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |
| `SupportsValidation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

