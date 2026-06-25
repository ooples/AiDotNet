---
title: "LARSSelector<T>"
description: "Least Angle Regression (LARS) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Regression`

Least Angle Regression (LARS) based Feature Selection.

## For Beginners

LARS is like a compromise between forward selection
and Lasso. It adds features gradually, moving in a direction that's equally
correlated with all currently active features. This gives a natural ordering
of feature importance.

## How It Works

Uses LARS algorithm to select features by progressively adding features
along the direction equiangular to all active features.

