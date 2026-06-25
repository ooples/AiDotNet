---
title: "WilcoxonSignedRank<T>"
description: "Wilcoxon Signed-Rank test for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

Wilcoxon Signed-Rank test for feature selection.

## For Beginners

When you have paired data (like measurements from
the same subjects at different times), this test checks if there's a consistent
difference. It doesn't assume normal distributions, making it more robust than
paired t-tests for real-world data.

## How It Works

The Wilcoxon Signed-Rank test is a non-parametric test for paired samples.
It tests whether the median difference between pairs of observations is zero,
useful for before/after comparisons or matched samples.

