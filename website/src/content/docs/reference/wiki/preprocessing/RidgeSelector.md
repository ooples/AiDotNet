---
title: "RidgeSelector<T>"
description: "Ridge Regression (L2) based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Ridge Regression (L2) based feature selection.

## For Beginners

Unlike Lasso (L1), Ridge doesn't set coefficients
exactly to zero. Instead, it shrinks all coefficients proportionally.
Features with larger coefficients after shrinking are more important.

## How It Works

Ridge regression adds L2 regularization to linear regression, shrinking
coefficients toward zero. Features with larger absolute coefficients
after regularization are considered more important.

