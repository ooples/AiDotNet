---
title: "KruskalWallisTest<T>"
description: "Kruskal-Wallis H test for non-parametric feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Kruskal-Wallis H test for non-parametric feature selection.

## For Beginners

Unlike ANOVA which assumes data follows a normal
bell curve, Kruskal-Wallis works with any data distribution. It converts
values to ranks and tests whether the ranks differ between groups. This is
more robust when your data has outliers or non-normal distributions.

## How It Works

A non-parametric alternative to one-way ANOVA that doesn't assume normal
distributions. Tests whether samples from different groups originate from
the same distribution based on ranks.

