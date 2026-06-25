---
title: "DistanceCorrelationSelector<T>"
description: "Distance Correlation based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Nonlinear`

Distance Correlation based Feature Selection.

## For Beginners

Distance correlation measures dependence between
variables using pairwise distances. Unlike Pearson correlation which only
detects linear relationships, distance correlation equals zero if and only if
the variables are truly independent. It ranges from 0 to 1.

## How It Works

Selects features based on distance correlation with the target,
which detects all types of dependence including nonlinear.

