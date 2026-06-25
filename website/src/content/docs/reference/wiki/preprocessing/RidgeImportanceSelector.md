---
title: "RidgeImportanceSelector<T>"
description: "Ridge Regression Importance based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Regression`

Ridge Regression Importance based Feature Selection.

## For Beginners

Ridge regression shrinks coefficients toward zero
but doesn't eliminate them. Features with larger coefficients after this
shrinkage are more important. Unlike Lasso, Ridge handles correlated features
gracefully by sharing importance among them.

## How It Works

Selects features based on their coefficient magnitudes from ridge regression,
which handles multicollinearity well while preserving all features.

