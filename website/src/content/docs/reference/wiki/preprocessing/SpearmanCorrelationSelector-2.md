---
title: "SpearmanCorrelationSelector<T>"
description: "Spearman Rank Correlation Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Spearman Rank Correlation Feature Selection.

## For Beginners

Spearman correlation uses ranks instead of actual
values. It converts each variable to ranks (1st, 2nd, 3rd, etc.) and then
computes the correlation of those ranks. This makes it robust to outliers
and able to detect non-linear but monotonic (always increasing or decreasing)
relationships.

## How It Works

Selects features based on their Spearman rank correlation with the target,
which measures monotonic relationships without assuming linearity.

