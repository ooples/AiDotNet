---
title: "CMIM<T>"
description: "Conditional Mutual Information Maximization (CMIM) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Conditional Mutual Information Maximization (CMIM) for feature selection.

## For Beginners

CMIM is cautious about redundancy. When considering a
new feature, it looks at how much information it provides about the target given
each already-selected feature (conditioning). It takes the worst case (minimum)
and picks the feature whose worst case is best. This ensures every selected feature
adds value no matter which other feature you consider.

## How It Works

CMIM uses a max-min criterion to select features. For each candidate, it computes
the minimum conditional mutual information with respect to all already-selected
features, then selects the candidate with the maximum of these minima.

