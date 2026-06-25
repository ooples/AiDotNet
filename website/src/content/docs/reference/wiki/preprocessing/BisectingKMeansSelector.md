---
title: "BisectingKMeansSelector<T>"
description: "Bisecting K-Means Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Bisecting K-Means Feature Selection.

## For Beginners

Bisecting K-means repeatedly splits the largest
cluster into two parts. We apply this to features (treating each feature as
a data point) and select representatives from different branches of the
resulting tree, ensuring diverse feature coverage.

## How It Works

Uses bisecting K-means clustering to hierarchically cluster features,
selecting representative features from each cluster branch.

