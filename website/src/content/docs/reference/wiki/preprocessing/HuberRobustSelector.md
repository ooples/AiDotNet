---
title: "HuberRobustSelector<T>"
description: "Huber Robust Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Robustness`

Huber Robust Feature Selection.

## For Beginners

The Huber loss combines squared error for small
residuals with absolute error for large ones. This makes the regression less
sensitive to outliers while still being efficient for normal data.

## How It Works

Selects features based on their coefficients in a Huber regression, which
downweights the influence of outliers using the Huber loss function.

