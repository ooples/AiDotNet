---
title: "PurgedKFoldSplitter<T>"
description: "Purged K-Fold cross-validation for time series with overlapping labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Purged K-Fold cross-validation for time series with overlapping labels.

## For Beginners

In financial time series, features often look into the future
(e.g., 5-day rolling average). This creates a problem: if your training data
includes day 95 and your test includes day 100, the rolling average for day 100
uses data from days 96-100, which includes your training period!

## How It Works

**The Solution - Purging:**
Remove (purge) samples around the test period that could leak information:

- Purge: Remove samples BEFORE test that could contaminate it
- Embargo: Remove samples AFTER test that could be affected by it

**When to Use:**

- Financial time series with overlapping windows
- Any time series where features depend on future values
- When you calculate rolling statistics

**Reference:** Based on Marcos López de Prado's methodology from
"Advances in Financial Machine Learning"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PurgedKFoldSplitter(Int32,Int32,Int32)` | Creates a new Purged K-Fold splitter for financial time series. |

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

