---
title: "WilcoxonSelector<T>"
description: "Wilcoxon Rank-Sum (Mann-Whitney U) test for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

Wilcoxon Rank-Sum (Mann-Whitney U) test for feature selection.

## For Beginners

Instead of comparing averages, this test ranks
all values and checks if one group tends to have higher ranks than the other.
It works even when data isn't normally distributed or has outliers.

## How It Works

The Wilcoxon test is a non-parametric alternative to the t-test. It compares
the ranks of values rather than the values themselves, making it robust to
outliers and non-normal distributions.

