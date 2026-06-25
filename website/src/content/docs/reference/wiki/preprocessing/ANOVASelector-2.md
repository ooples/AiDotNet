---
title: "ANOVASelector<T>"
description: "ANOVA F-Test based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

ANOVA F-Test based Feature Selection.

## For Beginners

ANOVA checks if groups differ significantly. Features
with high F-scores have very different values across classes, making them
useful for classification. Higher F means more class separation.

## How It Works

Selects features based on their ANOVA F-statistic, measuring the ratio of
between-group variance to within-group variance.

