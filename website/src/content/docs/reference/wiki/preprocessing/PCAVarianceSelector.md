---
title: "PCAVarianceSelector<T>"
description: "PCA Variance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Projection`

PCA Variance based Feature Selection.

## For Beginners

PCA finds the main directions of variation in data.
This selector identifies which original features contribute most to those
main directions, keeping features that explain the most variance.

## How It Works

Selects features based on their contribution to principal components,
measuring how much variance each feature contributes to the data.

