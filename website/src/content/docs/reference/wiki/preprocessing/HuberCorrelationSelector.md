---
title: "HuberCorrelationSelector<T>"
description: "Huber Correlation-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robust`

Huber Correlation-based Feature Selection.

## For Beginners

Regular correlation can be strongly affected by
outliers. The Huber method treats moderate deviations normally but reduces
the influence of extreme values. This gives you feature-target correlations
that better reflect the typical relationship, not just outlier effects.

## How It Works

Uses Huber M-estimator for robust correlation computation, downweighting
the influence of outliers in the feature-target relationship.

