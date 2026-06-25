---
title: "SpearmanCorrelation<T>"
description: "Spearman Rank Correlation-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Spearman Rank Correlation-based feature selection.

## For Beginners

Instead of measuring how well a straight line fits,
Spearman correlation checks if the feature and target increase together (or one
increases while the other decreases). It works with rankings, so it's robust to
outliers and can detect curved relationships that are still consistently going
up or down.

## How It Works

Spearman Correlation measures the monotonic relationship between features
and target using ranks instead of raw values. It can capture non-linear
monotonic relationships that Pearson correlation would miss.

