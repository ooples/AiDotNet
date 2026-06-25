---
title: "BimodalitySelector<T>"
description: "Bimodality Coefficient based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Moment`

Bimodality Coefficient based Feature Selection.

## For Beginners

Bimodality measures if data has two separate groups.
A high bimodality coefficient (>0.555 for uniform threshold) suggests the feature
naturally separates data into two clusters, which can be useful for classification.
The formula is: BC = (skewness² + 1) / (kurtosis + 3 × (n-1)²/((n-2)(n-3)))

## How It Works

Selects features based on their bimodality coefficient, which indicates
whether a distribution has two distinct modes (peaks).

