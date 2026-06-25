---
title: "SCAD<T>"
description: "Smoothly Clipped Absolute Deviation (SCAD) penalty for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Smoothly Clipped Absolute Deviation (SCAD) penalty for feature selection.

## For Beginners

LASSO tends to shrink large coefficients too much
(bias). SCAD fixes this by penalizing large coefficients less aggressively
while still shrinking small ones to zero. This gives you better estimates
of important feature effects.

## How It Works

SCAD is a non-convex penalty that addresses the bias problem of LASSO.
It applies the same penalty as LASSO for small coefficients but reduces
the penalty for large coefficients, leading to nearly unbiased estimates.

