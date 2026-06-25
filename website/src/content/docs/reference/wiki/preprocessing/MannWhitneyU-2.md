---
title: "MannWhitneyU<T>"
description: "Mann-Whitney U test for feature selection in binary classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Mann-Whitney U test for feature selection in binary classification.

## For Beginners

Mann-Whitney U is the non-parametric version of the t-test
for comparing two groups. It ranks all values and checks if one class has consistently
higher or lower ranks than the other. It's robust to outliers and doesn't assume
your data follows any particular distribution.

## How It Works

The Mann-Whitney U test is a non-parametric test that compares the rank distributions
of two groups. For binary classification, it tests whether one class tends to have
systematically higher values than the other for each feature.

