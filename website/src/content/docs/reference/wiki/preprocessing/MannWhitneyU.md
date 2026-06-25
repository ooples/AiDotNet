---
title: "MannWhitneyU<T>"
description: "Mann-Whitney U test for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

Mann-Whitney U test for feature selection.

## For Beginners

This test asks: "Do the values of this feature tend to
be higher in one group than another?" Unlike t-tests, it doesn't require the data
to follow a bell curve. It's great for data that's skewed or has outliers.

## How It Works

The Mann-Whitney U test is a non-parametric test that compares the distributions
of a feature between two groups. It doesn't assume normality, making it robust
for real-world data with outliers or skewed distributions.

