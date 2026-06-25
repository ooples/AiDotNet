---
title: "VarianceThreshold<T>"
description: "Variance Threshold feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Variance Threshold feature selection.

## For Beginners

If a feature has almost the same value for every
sample, it can't help distinguish between them. This method removes features
that are too "flat" or constant. A variance of 0 means the feature is exactly
the same for all samples.

## How It Works

Removes features whose variance is below a specified threshold. Features with
low variance contain little information and are unlikely to be predictive.

