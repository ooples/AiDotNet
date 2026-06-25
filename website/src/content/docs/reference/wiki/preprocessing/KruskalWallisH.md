---
title: "KruskalWallisH<T>"
description: "Kruskal-Wallis H test for non-parametric multi-class feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Kruskal-Wallis H test for non-parametric multi-class feature selection.

## For Beginners

This test extends Mann-Whitney U to more than 2 groups.
It ranks all values together, then checks if different classes have significantly
different rank distributions. Works well for non-normal data.

## How It Works

The Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA.
It tests whether samples from multiple groups come from the same distribution.

