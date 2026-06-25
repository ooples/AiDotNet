---
title: "LocallyWeightedRegressionOptions"
description: "Configuration options for Locally Weighted Regression, a non-parametric method that creates a model by fitting simple models to localized subsets of data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Locally Weighted Regression, a non-parametric method
that creates a model by fitting simple models to localized subsets of data.

## For Beginners

Locally Weighted Regression is like asking your neighbors for advice,
but giving more importance to those who live closest to you.

Imagine you're trying to estimate house prices:

- Traditional regression creates one formula for the entire city
- Locally Weighted Regression works differently - when estimating the price of a specific house,

it looks primarily at similar nearby houses

- Houses very similar to yours get a high "weight" (strong influence)
- Houses quite different from yours get a low "weight" (weak influence)
- This helps capture neighborhood-specific patterns that might get lost in a city-wide formula

This approach is particularly useful when different regions of your data behave differently,
as it can adapt to local patterns rather than forcing a single model on everything.
This class allows you to configure how the "neighborhood" is defined and how the local
models are calculated.

## How It Works

Locally Weighted Regression (LWR) is a memory-based technique that performs a regression
around a point of interest using only training data that are "local" to that point.
Unlike global methods that fit a single model to the entire dataset, LWR fits a separate
simple model for each query point by using nearby training examples weighted by their distance.
This allows the method to capture complex, non-linear relationships in the data without
specifying a global functional form.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bandwidth` | Gets or sets the bandwidth parameter that controls the size of the "neighborhood" used in locally weighted regression. |
| `DecompositionType` | Gets or sets the matrix decomposition method used to solve the weighted least squares problem at each query point. |
| `UseSoftMode` | Gets or sets whether to use soft (differentiable) mode for JIT compilation support. |

