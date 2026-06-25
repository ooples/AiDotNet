---
title: "FCBF<T>"
description: "Fast Correlation-Based Filter (FCBF) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Fast Correlation-Based Filter (FCBF) for feature selection.

## For Beginners

Having two features that say the same thing is wasteful.
FCBF first picks features that are useful for prediction, then removes features
that duplicate what other selected features already tell you. It's like building
a team where each member brings unique skills.

## How It Works

FCBF uses Symmetric Uncertainty to evaluate feature-target and feature-feature
correlations. It selects features that are highly correlated with the target
but not redundant with other selected features, using a fast backwards elimination
process.

