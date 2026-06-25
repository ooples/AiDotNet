---
title: "ElasticNetRegressionOptions<T>"
description: "Configuration options for Elastic Net Regression (combined L1 and L2 regularization)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Elastic Net Regression (combined L1 and L2 regularization).

## For Beginners

Elastic Net gives you the best of both Ridge and Lasso.

Lasso (L1) is great for feature selection but has a limitation: when features are
highly correlated, it tends to arbitrarily pick one and zero out the others.

Ridge (L2) handles correlated features well but doesn't do feature selection -
all features keep non-zero coefficients.

Elastic Net combines both:

- It can still set coefficients to zero (like Lasso) for feature selection
- It groups correlated features together (like Ridge) instead of picking one arbitrarily

When to use Elastic Net:

- When you have correlated features and want feature selection
- When Lasso's behavior on correlated features is problematic
- When you're not sure whether Ridge or Lasso is better

The l1_ratio parameter controls the mix:

- l1_ratio = 1.0: Pure Lasso (L1 only)
- l1_ratio = 0.0: Pure Ridge (L2 only)
- l1_ratio = 0.5: Equal mix of L1 and L2 (default)

Note: If your features are on different scales, consider normalizing your data
before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.

## How It Works

Elastic Net combines the penalties of Ridge (L2) and Lasso (L1) regression, providing
a balance between feature selection (from L1) and handling correlated features (from L2).

The objective function minimized is:
(1/2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||^2 / 2

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the overall regularization strength. |
| `L1Ratio` | Gets or sets the ratio of L1 penalty in the combined penalty. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the coordinate descent algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |
| `WarmStart` | Gets or sets whether to use warm starting for cross-validation. |

