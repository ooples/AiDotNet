---
title: "IsomapSelector<T>"
description: "Isomap-inspired Manifold-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Manifold`

Isomap-inspired Manifold-based Feature Selection.

## For Beginners

Data often lies on a curved surface (manifold).
This selector finds features that best preserve the true distances along
that surface, keeping features important for the data's intrinsic geometry.

## How It Works

Selects features that best preserve geodesic distances in the data manifold,
identifying features important for the underlying geometric structure.

