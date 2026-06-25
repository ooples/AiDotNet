---
title: "BaggingFeatureSelector<T>"
description: "Bagging-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Bagging-based Feature Selection.

## For Beginners

Bagging creates many random subsets of your data,
evaluates feature importance on each subset, then combines the results. This
helps ensure that the selected features are consistently important across
different parts of your data, not just important by chance.

## How It Works

Uses bootstrap aggregation (bagging) to create multiple subsamples and
aggregates feature importance across all subsamples for robust selection.

