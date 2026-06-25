---
title: "LassoSelector<T>"
description: "Lasso (L1) regularization-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Lasso (L1) regularization-based feature selection.

## For Beginners

Lasso is like a strict budget for your model. It
forces the model to "spend" wisely on features, often setting unimportant
feature weights to exactly zero. Features with non-zero weights after this
"budgeting" process are considered important.

## How It Works

Uses L1-regularized linear regression (Lasso) which drives some coefficients
exactly to zero, performing automatic feature selection. The regularization
parameter controls the sparsity of the solution.

