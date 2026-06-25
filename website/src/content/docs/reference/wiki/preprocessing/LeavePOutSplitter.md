---
title: "LeavePOutSplitter<T>"
description: "Leave-P-Out cross-validation where all combinations of p samples form the test sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation`

Leave-P-Out cross-validation where all combinations of p samples form the test sets.

## For Beginners

Leave-P-Out (LPO) tests all possible combinations of p samples.
This is an exhaustive evaluation but can be extremely expensive.

## How It Works

**Number of Splits:**
The number of splits is C(n, p) = n! / (p! × (n-p)!)

- n=10, p=2: 45 splits
- n=20, p=2: 190 splits
- n=10, p=3: 120 splits

**Warning:** The number of combinations grows very quickly!
LPO is only practical for very small datasets and small p values.

**When to Use:**

- Tiny datasets (<30 samples)
- When you need exhaustive evaluation
- Statistical research

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeavePOutSplitter(Int32)` | Creates a new Leave-P-Out cross-validation splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BinomialCoefficient(Int32,Int32)` | Calculates the binomial coefficient C(n, k) = n! / (k! × (n-k)!) |
| `GenerateCombinations(Int32,Int32)` | Generates all combinations of k items from n items. |
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

