---
title: "QuantileRegressionForestsOptions"
description: "Configuration options for Quantile Regression Forests, an extension of Random Forests that enables prediction of conditional quantiles rather than just the conditional mean."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Quantile Regression Forests, an extension of Random Forests that enables
prediction of conditional quantiles rather than just the conditional mean.

## For Beginners

Quantile Regression Forests help predict not just a single value, but a range of possible values with their probabilities.

Think about weather forecasting:

- A regular forecast might say "tomorrow's temperature will be 75°F"
- But Quantile Regression Forests could tell you:
- "There's a 10% chance it will be below 70°F"
- "There's a 50% chance it will be below 75°F" (the median)
- "There's a 90% chance it will be below 80°F"

What this algorithm does:

- It builds many decision trees, just like a regular Random Forest
- But instead of averaging their predictions to get a single answer
- It keeps track of all possible outcomes and their distributions
- This lets you understand the uncertainty in your predictions

This is especially useful when:

- You need to know the range of possible outcomes, not just the average
- Your data has varying levels of uncertainty in different regions
- The distribution of possible outcomes is not symmetric
- Risk assessment is as important as the prediction itself

For example, in financial forecasting, knowing there's a 5% chance of losing $10,000
is very different information than just knowing the average expected return.

This class lets you configure how the forest of trees is built and processed.

## How It Works

Quantile Regression Forests extend the Random Forest algorithm to provide full conditional distributions
instead of just point estimates. While standard Random Forests estimate the conditional mean E(Y|X),
Quantile Regression Forests can estimate any conditional quantile Q(a|X) for a ? (0,1), including
medians and prediction intervals. This is achieved by keeping track of all target values in the leaf
nodes of each tree, rather than just their averages. The algorithm provides a non-parametric way to
estimate conditional distributions, making it particularly valuable for problems where uncertainty
quantification is important or where the conditional distribution is non-Gaussian, skewed, or
heteroscedastic (having non-constant variance). Quantile Regression Forests inherit many advantages of
Random Forests, including handling of non-linear relationships, robustness to outliers, and minimal
parameter tuning requirements.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDegreeOfParallelism` | Gets or sets the maximum degree of parallelism for tree building. |
| `NumberOfTrees` | Gets or sets the number of trees to grow in the forest. |

