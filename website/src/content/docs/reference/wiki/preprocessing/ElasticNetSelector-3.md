---
title: "ElasticNetSelector<T>"
description: "Elastic Net based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sparsity`

Elastic Net based Feature Selection.

## For Beginners

Elastic Net combines Lasso (L1) and Ridge (L2)
penalties. L1 selects features by zeroing coefficients; L2 groups correlated
features together. The mix gives stable feature selection even when features
are correlated, which pure Lasso struggles with.

## How It Works

Selects features using Elastic Net regularization, which combines
L1 and L2 penalties for better stability with correlated features.

