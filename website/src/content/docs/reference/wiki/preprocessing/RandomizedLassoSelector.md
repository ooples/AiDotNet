---
title: "RandomizedLassoSelector<T>"
description: "Randomized Lasso Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Stability`

Randomized Lasso Feature Selection.

## For Beginners

Regular Lasso can be unstable - small changes
in data can lead to different features being selected. Randomized Lasso
runs Lasso many times with random perturbations and selects features
that are consistently chosen, giving more reliable results.

## How It Works

Combines Lasso with random reweighting and subsampling to provide
more stable feature selection that is less sensitive to the choice
of regularization parameter.

