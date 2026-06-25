---
title: "KruskalWallis<T>"
description: "Kruskal-Wallis H-test for non-parametric feature selection in classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Kruskal-Wallis H-test for non-parametric feature selection in classification.

## For Beginners

While ANOVA assumes your data follows a bell curve,
Kruskal-Wallis makes no such assumption. It ranks all values and checks if different
classes tend to have systematically higher or lower ranks. It's more robust when
your data has outliers or isn't normally distributed.

## How It Works

The Kruskal-Wallis test is a non-parametric version of one-way ANOVA. It compares
the rank distributions across classes rather than assuming normal distributions.
Features that lead to different rank distributions across classes are considered important.

