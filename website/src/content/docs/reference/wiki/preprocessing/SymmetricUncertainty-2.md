---
title: "SymmetricUncertainty<T>"
description: "Symmetric Uncertainty Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Symmetric Uncertainty Feature Selection.

## For Beginners

Think of this as measuring how well knowing
one value helps predict another. Unlike regular correlation, it works
for any type of relationship (not just linear) and always gives a value
between 0 and 1, making it easy to compare different features.

## How It Works

Symmetric Uncertainty is a normalized version of mutual information that
measures the correlation between features and targets. It ranges from 0
(no correlation) to 1 (perfect correlation) and is symmetric.

