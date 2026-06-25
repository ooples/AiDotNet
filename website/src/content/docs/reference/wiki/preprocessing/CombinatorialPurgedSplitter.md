---
title: "CombinatorialPurgedSplitter<T>"
description: "Combinatorial Purged Cross-Validation splitter for time series data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Combinatorial Purged Cross-Validation splitter for time series data.

## For Beginners

This is an advanced cross-validation method designed specifically
for financial time series where data leakage can invalidate backtests.

## How It Works

**How It Works:**

1. Divide data into n groups based on time periods
2. Generate all combinations of k groups for testing
3. For each combination, remaining groups form training set
4. Apply purging (remove samples near test boundaries) and embargo (block after test)

**Use Cases:**

- Financial backtesting where overlapping data causes leakage
- Time series with autocorrelation that persists across samples
- Generating many independent train/test splits from limited time series

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CombinatorialPurgedSplitter(Int32,Int32,Int32,Int32,Int32)` | Creates a new Combinatorial Purged CV splitter. |

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

