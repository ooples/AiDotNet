---
title: "PearsonCorrelation<T>"
description: "Pearson Correlation for feature selection based on linear relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Pearson Correlation for feature selection based on linear relationships.

## For Beginners

Pearson correlation tells you how well a straight line
can describe the relationship between a feature and the target. If r is close to 1
or -1, the feature moves predictably with the target (up together or opposite).
If r is near 0, there's no linear relationship (though there could be a curved one).

## How It Works

Measures the linear relationship between features and target using Pearson's r.
Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).

