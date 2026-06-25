---
title: "VarianceInflationSelector<T>"
description: "Variance Inflation Factor (VIF) Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Variance Inflation Factor (VIF) Feature Selection.

## For Beginners

VIF measures how much a feature can be predicted
by other features. A high VIF means that feature is highly correlated with
others (redundant). We remove features with high VIF to keep only the
independent, non-redundant ones.

## How It Works

Removes features with high multicollinearity by computing the Variance
Inflation Factor for each feature and removing those above a threshold.

