---
title: "HorseshoeSelector<T>"
description: "Horseshoe Prior Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bayesian`

Horseshoe Prior Feature Selection.

## For Beginners

The horseshoe prior is shaped like a horseshoe -
most feature weights get pushed to zero (irrelevant), but truly important
features are allowed to have large weights. It's very good at finding
sparse solutions where only a few features matter.

## How It Works

Uses the horseshoe prior which has heavy tails (allowing large coefficients
for important features) and infinite mass at zero (shrinking irrelevant
features to exactly zero).

