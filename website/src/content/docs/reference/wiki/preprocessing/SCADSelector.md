---
title: "SCADSelector<T>"
description: "Smoothly Clipped Absolute Deviation (SCAD) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Smoothly Clipped Absolute Deviation (SCAD) for feature selection.

## For Beginners

SCAD is like an improved version of Lasso. Lasso
tends to shrink all coefficients, even important ones. SCAD lets truly important
features keep their full strength while still zeroing out unimportant ones.
This gives you better feature selection with less bias.

## How It Works

SCAD is a nonconvex penalty that reduces bias for large coefficients while
maintaining sparsity. Unlike Lasso, SCAD doesn't overshrink large coefficients,
providing near-unbiased estimates for significant features.

