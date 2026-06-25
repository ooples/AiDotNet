---
title: "DistanceCorrelationSelector<T>"
description: "Distance Correlation based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Nonparametric`

Distance Correlation based Feature Selection.

## For Beginners

Unlike Pearson correlation that only finds linear
relationships, distance correlation can find any type of relationship. If two
variables are truly independent, distance correlation will be zero, but it can
also detect complex nonlinear patterns.

## How It Works

Selects features based on distance correlation with the target, which can
detect both linear and nonlinear dependencies between variables.

