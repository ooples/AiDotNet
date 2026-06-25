---
title: "DispersionRatio<T>"
description: "Dispersion Ratio for comparing within-class to between-class variation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Dispersion Ratio for comparing within-class to between-class variation.

## For Beginners

A good feature for classification should have class
groups that are far apart (high between-class variance) and tight clusters
(low within-class variance). The dispersion ratio captures this by dividing
the separation by the spread. High values mean easy-to-separate classes.

## How It Works

The dispersion ratio measures the ratio of between-class variance to within-class
variance. Features with high dispersion ratios have well-separated class means
relative to the variation within each class.

