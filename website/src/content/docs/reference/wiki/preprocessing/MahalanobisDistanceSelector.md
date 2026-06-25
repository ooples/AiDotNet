---
title: "MahalanobisDistanceSelector<T>"
description: "Mahalanobis Distance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distance`

Mahalanobis Distance based Feature Selection.

## For Beginners

Mahalanobis distance is like Euclidean distance but
adjusted for how data is spread out. It's useful when features are correlated
or have different scales. Features that contribute more to class separation
considering the data's shape are selected.

## How It Works

Selects features based on the Mahalanobis distance contribution, which accounts
for correlations between features and class-specific covariances.

