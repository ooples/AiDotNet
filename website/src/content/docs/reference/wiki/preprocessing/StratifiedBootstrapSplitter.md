---
title: "StratifiedBootstrapSplitter<T>"
description: "Stratified bootstrap sampling that preserves class distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap`

Stratified bootstrap sampling that preserves class distribution.

## For Beginners

This combines bootstrap sampling with stratification.
Within each class, we sample with replacement, ensuring the bootstrap sample
maintains approximately the same class proportions as the original data.

## How It Works

**When to Use:**

- Bootstrap with imbalanced classes
- When you need class proportions preserved in each bootstrap sample

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedBootstrapSplitter(Int32,Int32)` | Creates a new stratified bootstrap splitter. |

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

