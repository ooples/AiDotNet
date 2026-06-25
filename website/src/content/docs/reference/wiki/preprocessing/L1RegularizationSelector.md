---
title: "L1RegularizationSelector<T>"
description: "L1 Regularization (Lasso-like) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparsity`

L1 Regularization (Lasso-like) based Feature Selection.

## For Beginners

L1 regularization adds a penalty proportional
to the absolute value of coefficients. This penalty pushes weak coefficients
to exactly zero, effectively selecting important features. The stronger the
regularization, the fewer features are selected.

## How It Works

Selects features using L1-regularized regression (Lasso), which
automatically sets unimportant feature coefficients to zero.

