---
title: "PCABasedSelection<T>"
description: "PCA-Based Feature Selection using loadings analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Dimensionality`

PCA-Based Feature Selection using loadings analysis.

## For Beginners

PCA finds combinations of features that capture
the most variation in your data. By looking at which original features
contribute most to these combinations, we can identify which features
are most informative overall.

## How It Works

Uses Principal Component Analysis to identify which original features
contribute most to the principal components. Features with high loadings
on important components are selected.

