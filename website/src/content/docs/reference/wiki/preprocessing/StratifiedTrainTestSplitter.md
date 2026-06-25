---
title: "StratifiedTrainTestSplitter<T>"
description: "Stratified train/test split that preserves class distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified`

Stratified train/test split that preserves class distribution.

## For Beginners

A stratified split ensures that both your training and test sets
have the same proportion of each class as the original data.

## How It Works

**Why This Matters:**
Imagine your data has 90% cats and 10% dogs. With a random split, you might get
unlucky and have 95% cats in training but only 70% cats in test. This would
make your model evaluation unreliable. Stratification prevents this.

**Industry Standard:** For classification tasks, ALWAYS use stratified splitting
unless you have a specific reason not to.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedTrainTestSplitter(Double,Boolean,Int32)` | Creates a new stratified train/test splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

