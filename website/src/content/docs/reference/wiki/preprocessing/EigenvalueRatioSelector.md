---
title: "EigenvalueRatioSelector<T>"
description: "Eigenvalue Ratio based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Spectral`

Eigenvalue Ratio based Feature Selection.

## For Beginners

Eigenvalues measure how much variance each
direction in data captures. This selector keeps features that contribute
most to the dominant directions of variance in your data.

## How It Works

Selects features based on the ratio of top eigenvalues when features are
added, measuring how much each feature contributes to data variance.

