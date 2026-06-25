---
title: "KendallTauSelector<T>"
description: "Kendall's Tau Correlation based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Nonparametric`

Kendall's Tau Correlation based Feature Selection.

## For Beginners

Kendall's tau measures how often pairs of data
points are in the same order for both feature and target. It's robust to
outliers and doesn't assume a linear relationship, just that both variables
tend to increase or decrease together.

## How It Works

Selects features based on Kendall's tau rank correlation coefficient with
the target, which measures ordinal association between features and target.

