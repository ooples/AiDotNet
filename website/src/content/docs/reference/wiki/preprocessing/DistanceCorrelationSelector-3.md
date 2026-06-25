---
title: "DistanceCorrelationSelector<T>"
description: "Distance Correlation Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Distance Correlation Feature Selection.

## For Beginners

Distance correlation is like regular correlation
but can detect any type of relationship, not just linear ones. Unlike regular
correlation, a distance correlation of 0 truly means no relationship exists.
It computes distances between all pairs of points and correlates those distances.

## How It Works

Uses distance correlation to detect both linear and non-linear dependencies
between features and the target. Distance correlation is zero if and only
if the variables are independent.

