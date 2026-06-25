---
title: "GradientBoostingRegressionOptions"
description: "Configuration options for Gradient Boosting Regression, an ensemble learning technique that combines multiple decision trees to create a powerful regression model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Gradient Boosting Regression, an ensemble learning technique that combines
multiple decision trees to create a powerful regression model.

## For Beginners

Think of Gradient Boosting as a team of decision trees working together to
make predictions. Instead of relying on just one tree (which might make mistakes), gradient boosting
builds trees one after another, with each new tree focusing specifically on correcting the mistakes
made by all the previous trees.

Imagine you're trying to predict house prices. The first tree might make rough predictions, getting
some houses right but being way off on others. The second tree doesn't try to predict the full house
price again - instead, it specifically focuses on the houses where the first tree was wrong, trying to
predict the error. This process continues, with each new tree fixing more subtle mistakes, until you
have a collection of trees that work together to make very accurate predictions.

Gradient boosting models are among the most powerful and widely used machine learning algorithms,
especially for structured/tabular data, because they often achieve excellent accuracy while being
relatively easy to use.

## How It Works

Gradient Boosting is an ensemble machine learning technique that builds multiple decision trees
sequentially, with each tree correcting the errors made by the previous trees. This approach typically
produces more accurate models than single decision trees, at the cost of increased complexity and
training time. This class inherits from DecisionTreeOptions, so all options for configuring individual
trees are also available.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate (shrinkage) applied to each tree's contribution. |
| `NumberOfTrees` | Gets or sets the number of trees (estimators) in the ensemble. |
| `SubsampleRatio` | Gets or sets the fraction of samples used for fitting each tree. |

