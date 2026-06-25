---
title: "CoefficientOfVariation<T>"
description: "Coefficient of Variation (CV) for unsupervised feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Coefficient of Variation (CV) for unsupervised feature selection.

## For Beginners

Some features barely change across samples (low CV),
while others vary a lot (high CV). Features that don't change much probably
can't distinguish between different outcomes. CV is useful when features are
on different scales, because it measures relative variation.

## How It Works

The coefficient of variation is the ratio of standard deviation to mean,
expressed as a percentage. Features with low CV have relatively little
variation and may not be informative.

