---
title: "EnsembleFeatureSelector<T>"
description: "Ensemble Feature Selection combining multiple selection methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Ensemble Feature Selection combining multiple selection methods.

## For Beginners

Different selection methods have different strengths.
By combining them, we get more robust and reliable feature selection. Features
that multiple methods agree on are more trustworthy.

## How It Works

Ensemble Feature Selection aggregates results from multiple feature selection
methods. Features consistently selected across methods are more likely to be
truly important.

Aggregation methods:

- Voting: Select features chosen by majority of methods
- Ranking: Aggregate rankings using Borda count or similar
- Weighted: Weight methods by their reliability

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSelector(Func<Matrix<>,Vector<>,Int32[]>)` | Adds a selector function to the ensemble. |

