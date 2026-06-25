---
title: "CorrelationFilter<T>"
description: "Correlation Filter for removing highly correlated features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Correlation Filter for removing highly correlated features.

## For Beginners

If two features are very similar (highly correlated),
keeping both is redundant. This filter finds such pairs and keeps only one,
reducing data size without losing much information.

## How It Works

Identifies pairs of features with correlation above a threshold and removes
one from each pair. This reduces multicollinearity and feature redundancy.

