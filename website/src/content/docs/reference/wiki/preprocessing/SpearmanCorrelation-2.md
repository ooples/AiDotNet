---
title: "SpearmanCorrelation<T>"
description: "Spearman rank correlation for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Spearman rank correlation for feature selection.

## For Beginners

While Pearson correlation only detects linear relationships,
Spearman detects any monotonic relationship (consistently increasing or decreasing).
It works by ranking values, so outliers have less impact. Perfect for ordinal data
or when relationships are curved but consistent.

## How It Works

Measures the monotonic relationship between features and target using ranks
instead of raw values. Robust to outliers and non-linear monotonic relationships.

