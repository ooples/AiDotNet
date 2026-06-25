---
title: "TrainValTestSplitter<T>"
description: "Three-way splitter that divides data into training, validation, and test sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Basic`

Three-way splitter that divides data into training, validation, and test sets.

## For Beginners

This splitter creates three separate sets from your data:

- **Training set:** Data your model learns from (~70%)
- **Validation set:** Data used to tune hyperparameters and prevent overfitting (~15%)
- **Test set:** Data for final evaluation - never touch during training (~15%)

## How It Works

**Why Three Sets?**
If you only have train/test, you might tune your model to perform well on the test set,
which means you're "cheating" - your test set is no longer truly unseen data.
The validation set lets you tune without contaminating your final test evaluation.

**Industry Standard:** 70/15/15 or 60/20/20 splits are common.

**When to Use:**

- Medium to large datasets
- When you need to tune hyperparameters
- Deep learning (where validation is essential for early stopping)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainValTestSplitter(Double,Double,Boolean,Int32)` | Creates a new train/validation/test splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `SupportsValidation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `SplitIndicesOnly(Int32,Vector<>)` |  |

