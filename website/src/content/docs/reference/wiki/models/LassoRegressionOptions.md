---
title: "LassoRegressionOptions<T>"
description: "Configuration options for Lasso Regression (L1 regularized linear regression)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Lasso Regression (L1 regularized linear regression).

## For Beginners

Lasso Regression is like Ridge Regression but with a key difference:
it can completely eliminate unimportant features.

While Ridge Regression shrinks coefficients toward zero but never quite reaches it,
Lasso can set coefficients exactly to zero. This makes Lasso useful for:

- Feature selection: Identifying which features actually matter
- Sparse models: Creating simpler models with fewer non-zero coefficients
- High-dimensional data: When you have many features but suspect only a few are relevant

Example scenario:

- You have 100 features for predicting house prices
- Only 10 of them actually matter (location, size, etc.)
- Lasso will automatically set the other 90 coefficients to zero
- This gives you a simpler, more interpretable model

The trade-off:

- Lasso requires iterative optimization (slower than Ridge)
- If features are highly correlated, Lasso might arbitrarily pick one and zero out others
- For groups of correlated features, consider ElasticNet instead

Note: If your features are on different scales, consider normalizing your data
before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.

## How It Works

Lasso (Least Absolute Shrinkage and Selection Operator) Regression extends ordinary least squares
by adding an L1 penalty term to the loss function. Unlike Ridge Regression (L2), Lasso can shrink
coefficients exactly to zero, effectively performing automatic feature selection.

The objective function minimized is: (1/2n) * ||y - Xw||^2 + alpha * ||w||_1
where alpha controls the strength of the regularization and ||w||_1 is the L1 norm (sum of absolute values).

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the regularization strength (alpha). |
| `MaxIterations` | Gets or sets the maximum number of iterations for the coordinate descent algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |
| `WarmStart` | Gets or sets whether to use warm starting for cross-validation. |

