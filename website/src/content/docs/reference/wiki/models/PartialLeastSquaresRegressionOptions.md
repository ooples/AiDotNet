---
title: "PartialLeastSquaresRegressionOptions<T>"
description: "Configuration options for Partial Least Squares Regression (PLS), a technique that combines features of principal component analysis and multiple regression to handle multicollinearity and high-dimensional data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Partial Least Squares Regression (PLS), a technique that combines
features of principal component analysis and multiple regression to handle multicollinearity
and high-dimensional data.

## For Beginners

Partial Least Squares Regression is a special technique for finding relationships in complex data.

Imagine you're trying to predict house prices based on 50 different measurements:

- Many of these measurements might be related (like square footage and number of rooms)
- With so many variables, traditional regression methods might struggle

What PLS does:

- Instead of using all 50 original measurements directly
- It creates a smaller set of "super measurements" (called components)
- Each component combines multiple original measurements in a smart way
- These components capture the most important patterns relevant to your prediction

Think of it like cooking:

- Traditional regression uses each ingredient separately
- PLS creates a few key flavor profiles (combining multiple ingredients)
- Then uses these flavor profiles to create the final dish

This approach is especially useful when:

- You have more variables than data points
- Your variables are highly correlated with each other
- You need to reduce noise and focus on the most important patterns

This class lets you configure how many of these "super measurements" (components)
to use in your analysis.

## How It Works

Partial Least Squares Regression is a powerful statistical method that finds a linear regression
model by projecting both the predictor variables (X) and the response variables (Y) to a new space.
Unlike ordinary least squares regression, which can struggle with multicollinearity (highly correlated
predictors) and high-dimensional data, PLS creates latent variables (components) that maximize the
covariance between X and Y. This approach is particularly valuable in situations where the number of
predictor variables exceeds the number of observations, or when predictors are highly correlated.
PLS has found wide application in fields such as chemometrics, spectroscopy, and bioinformatics.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumComponents` | Gets or sets the number of latent components to extract in the PLS model. |

