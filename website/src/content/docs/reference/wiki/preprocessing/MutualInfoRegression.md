---
title: "MutualInfoRegression<T>"
description: "Mutual Information for regression-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Regression`

Mutual Information for regression-based feature selection.

## For Beginners

While correlation only finds straight-line relationships,
mutual information can detect any pattern where knowing one value tells you about
another. This is useful for finding features with curved or complex relationships
to your target that linear methods would miss.

## How It Works

Mutual Information Regression estimates the mutual information between each
continuous feature and a continuous target. Unlike correlation, MI can capture
non-linear dependencies between variables.

