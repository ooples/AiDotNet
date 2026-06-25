---
title: "GeneralizedAdditiveModelOptions<T>"
description: "Configuration options for Generalized Additive Models (GAMs), which are flexible regression models that combine multiple simple functions to model complex relationships."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Generalized Additive Models (GAMs), which are flexible regression models
that combine multiple simple functions to model complex relationships.

## For Beginners

Think of a Generalized Additive Model as a more flexible version of linear
regression. In linear regression, you assume each input feature has a straight-line relationship with
your target variable (like y = mx + b from high school math). GAMs relax this assumption by allowing
each feature to have its own curved relationship with the target. 

For example, if you're predicting house prices, a linear model might assume that price increases by a
fixed amount for each additional square foot. A GAM could capture that small houses might see a bigger
price increase per square foot than very large houses, where additional space might add less value.

GAMs achieve this flexibility while still being relatively easy to interpret compared to "black box"
models like neural networks, because you can visualize how each individual feature affects the prediction.
They're a good middle ground between simple linear models and complex machine learning algorithms.

## How It Works

Generalized Additive Models extend linear regression by allowing non-linear relationships between
predictors and the target variable. They work by fitting a separate smooth function (typically a spline)
for each predictor variable and then adding these functions together. This approach maintains much of
the interpretability of linear models while allowing for more flexible relationships.

## Properties

| Property | Summary |
|:-----|:--------|
| `Degree` | Gets or sets the degree of the polynomial splines used in the model. |
| `NumSplines` | Gets or sets the number of spline basis functions to use for each feature. |

