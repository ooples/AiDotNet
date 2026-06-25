---
title: "BalancedSplitter<T>"
description: "Balanced splitter that ensures equal representation of each class in train and test sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified`

Balanced splitter that ensures equal representation of each class in train and test sets.

## For Beginners

When you have imbalanced classes (e.g., 90% normal, 10% fraud),
this splitter ensures both train and test sets have equal numbers of each class
by undersampling the majority classes.

## How It Works

**How It Works:**

1. Find the smallest class size
2. Sample equally from each class to match the smallest
3. Split the balanced data into train/test

**When to Use:**

- When you want to evaluate model performance without class imbalance effects
- When minority class performance is critical
- For initial model development before handling imbalance in other ways

**Caution:** This discards data from majority classes. For production,
consider SMOTE or other oversampling techniques instead.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BalancedSplitter(Double,Boolean,Int32)` | Creates a new balanced splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

