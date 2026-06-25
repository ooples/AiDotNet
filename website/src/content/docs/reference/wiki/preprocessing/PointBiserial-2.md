---
title: "PointBiserial<T>"
description: "Point-Biserial Correlation for feature selection with binary target."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Point-Biserial Correlation for feature selection with binary target.

## For Beginners

When your target is binary (like pass/fail, yes/no),
point-biserial correlation tells you how much a continuous feature differs between
the two groups. If the feature values are very different for class 0 vs class 1,
the correlation is high and the feature is useful for classification.

## How It Works

Point-biserial correlation measures the relationship between a continuous feature
and a binary (dichotomous) target. It's mathematically equivalent to Pearson
correlation when one variable is binary.

