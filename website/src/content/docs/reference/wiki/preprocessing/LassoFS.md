---
title: "LassoFS<T>"
description: "LASSO (Least Absolute Shrinkage and Selection Operator) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

LASSO (Least Absolute Shrinkage and Selection Operator) for feature selection.

## For Beginners

LASSO is a special kind of linear regression that
automatically picks only the most important features. It does this by adding
a penalty that forces unimportant feature weights to become exactly zero,
effectively removing them from the model.

## How It Works

LASSO uses L1 regularization during linear regression training, which naturally
drives some coefficients to exactly zero. Features with non-zero coefficients
are selected for the final model.

