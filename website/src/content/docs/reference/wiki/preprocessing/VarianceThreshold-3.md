---
title: "VarianceThreshold<T>"
description: "Variance Threshold for removing low-variance features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Variance Threshold for removing low-variance features.

## For Beginners

If a feature has the same value for almost all
samples, it can't help distinguish between them. This method removes such
uninformative features automatically.

## How It Works

Removes features with variance below a specified threshold. Features with
very low variance carry little information and are often constant or
near-constant.

