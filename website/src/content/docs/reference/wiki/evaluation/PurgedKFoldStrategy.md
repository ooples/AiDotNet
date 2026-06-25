---
title: "PurgedKFoldStrategy<T>"
description: "Purged K-Fold: K-Fold with temporal purging to prevent data leakage in financial/time-dependent data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Purged K-Fold: K-Fold with temporal purging to prevent data leakage in financial/time-dependent data.

## For Beginners

Purged K-Fold adds a gap between training and test sets to prevent
temporal data leakage:

- Standard K-Fold can leak information when observations overlap in time
- Purging removes training samples that are temporally close to test samples
- Essential for financial data where future information must not influence past predictions

## How It Works

**Example:** If you're predicting stock returns using 5-day windows:

- Test period: Day 100-110
- Without purging: Training might include days 95-99 (overlapping windows!)
- With purging: Training excludes days 95-114 (5-day buffer on each side)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PurgedKFoldStrategy(Int32,Int32,Int32[])` | Initializes Purged K-Fold cross-validation. |

