---
title: "FCBF<T>"
description: "Fast Correlation-Based Filter (FCBF) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Fast Correlation-Based Filter (FCBF) for feature selection.

## For Beginners

FCBF finds features that are strongly related to
the target and removes any feature that's "dominated" by another (meaning
the other feature is just as good at predicting the target AND is more
correlated with this feature). This efficiently eliminates redundancy.

## How It Works

FCBF uses symmetrical uncertainty to identify relevant features and remove
redundant ones. It's efficient for high-dimensional data with the concept
of predominant features that are not dominated by any other feature.

