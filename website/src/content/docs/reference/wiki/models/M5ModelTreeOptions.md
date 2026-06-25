---
title: "M5ModelTreeOptions"
description: "Configuration options for the M5 Model Tree algorithm, which combines decision trees with linear regression models at the leaf nodes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the M5 Model Tree algorithm, which combines decision trees
with linear regression models at the leaf nodes.

## For Beginners

The M5 Model Tree is a powerful algorithm that combines two approaches
to make predictions about continuous values (like prices, temperatures, or heights).

Imagine you're trying to predict house prices:

- A regular decision tree would divide houses into categories (like "small houses in good neighborhoods")

and assign an average price to each category

- The M5 Model Tree does something smarter: it divides the houses into categories, but then creates

a custom formula for each category

Think of it like this:

- First, it groups similar houses together (like a traditional decision tree)
- Then, within each group, it creates a formula that considers factors like exact square footage,

number of bathrooms, etc. (using linear regression)

- This gives you more precise predictions than a simple average for each group

This class allows you to configure how the tree is built, pruned, and how its predictions are smoothed.

## How It Works

The M5 Model Tree is an extension of decision trees for regression problems, originally proposed by
Quinlan. Unlike traditional regression trees that store a constant value at each leaf, M5 Model Trees
fit a multivariate linear regression model at each leaf node. This combination allows the algorithm to
model both non-linear and linear relationships in the data efficiently. The algorithm includes pruning
mechanisms to prevent overfitting and smoothing techniques to improve predictions at the boundaries
between different linear models.

## Properties

| Property | Summary |
|:-----|:--------|
| `MinInstancesPerLeaf` | Gets or sets the minimum number of training instances required at each leaf node. |
| `PruningFactor` | Gets or sets the pruning factor that controls the trade-off between model complexity and error. |
| `SmoothingConstant` | Gets or sets the smoothing constant that controls the blending of predictions across different models in the tree. |
| `UseLinearRegressionAtLeaves` | Gets or sets whether to use linear regression models at leaf nodes instead of constant values. |
| `UsePruning` | Gets or sets whether to apply pruning to the tree after it is fully grown. |

