---
title: "ElasticNetSelector<T>"
description: "Elastic Net Regularization based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Regression`

Elastic Net Regularization based Feature Selection.

## For Beginners

Elastic Net combines Lasso (which sets unimportant
features to zero) and Ridge (which handles correlated features well). Features
that survive this double penalty are likely important.

## How It Works

Uses Elastic Net regularization (combination of L1 and L2 penalties) to select
features by identifying those with non-zero coefficients after regularized regression.

