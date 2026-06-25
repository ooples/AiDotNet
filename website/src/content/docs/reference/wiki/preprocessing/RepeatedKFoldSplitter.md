---
title: "RepeatedKFoldSplitter<T>"
description: "Repeated K-Fold cross-validation that runs K-Fold multiple times with different random seeds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Repeated K-Fold cross-validation that runs K-Fold multiple times with different random seeds.

## For Beginners

Repeated K-Fold runs K-Fold cross-validation multiple times,
each time with a different random shuffle. This gives even more stable performance estimates.

## How It Works

**How It Works:**

**When to Use:**

- When you need very reliable performance estimates
- For statistical significance testing
- When comparing models and small differences matter

**Cost:** Runs k × n_repeats evaluations, so it's n_repeats times slower than regular K-Fold.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepeatedKFoldSplitter(Int32,Int32,Int32)` | Creates a new Repeated K-Fold cross-validation splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

