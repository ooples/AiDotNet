---
title: "ManhattanDistanceSelector<T>"
description: "Manhattan Distance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distance`

Manhattan Distance based Feature Selection.

## For Beginners

Manhattan distance measures distance by only moving
along axes (like city blocks). It's less sensitive to outliers than Euclidean
distance and useful when features have different scales.

## How It Works

Selects features that maximize class separation based on Manhattan (L1) distance
between class centroids.

