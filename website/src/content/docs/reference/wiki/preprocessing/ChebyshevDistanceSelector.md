---
title: "ChebyshevDistanceSelector<T>"
description: "Chebyshev Distance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distance`

Chebyshev Distance based Feature Selection.

## For Beginners

Chebyshev distance is the maximum difference in any
single dimension. It's useful when you care about the worst-case difference
between classes. Features with large maximum separations are preferred.

## How It Works

Selects features that maximize class separation based on Chebyshev (L-infinity)
distance between class centroids.

