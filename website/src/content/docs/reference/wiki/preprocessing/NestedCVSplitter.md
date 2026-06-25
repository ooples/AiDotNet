---
title: "NestedCVSplitter<T>"
description: "Nested (Double) Cross-Validation splitter for unbiased model selection and evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Nested`

Nested (Double) Cross-Validation splitter for unbiased model selection and evaluation.

## For Beginners

Regular cross-validation can give biased results if you use it
for both model selection (hyperparameter tuning) AND performance estimation.
Nested CV solves this with two levels of cross-validation.

## How It Works

**How It Works:**

- Outer loop: Splits data into train+val and test for final evaluation
- Inner loop: Splits train+val for model selection/hyperparameter tuning

**Visual Example (3-outer, 2-inner):**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NestedCVSplitter(Int32,Int32,Boolean,Int32)` | Creates a new Nested CV splitter. |

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

