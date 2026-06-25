---
title: "DistanceCorrelation<T>"
description: "Distance Correlation-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Distance Correlation-based feature selection.

## For Beginners

Regular correlation only catches straight-line
relationships. Distance correlation can detect curved, U-shaped, or any type
of pattern between variables. A distance correlation of 0 means truly no
relationship at all.

## How It Works

Distance correlation measures the dependence between random variables, detecting
both linear and nonlinear relationships. Unlike Pearson correlation, it equals
zero if and only if the variables are independent.

