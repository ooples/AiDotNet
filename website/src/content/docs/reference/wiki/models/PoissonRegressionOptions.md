---
title: "PoissonRegressionOptions<T>"
description: "Configuration options for Poisson Regression, a specialized form of regression analysis used for modeling count data and contingency tables where the dependent variable consists of non-negative integers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Poisson Regression, a specialized form of regression analysis used for modeling
count data and contingency tables where the dependent variable consists of non-negative integers.

## For Beginners

Poisson Regression is a technique specially designed for predicting counts of things.

Imagine you want to predict:

- How many customers will visit a store each hour
- How many calls a support center will receive each day
- How many defects will appear in manufactured products

When working with counts:

- You never have negative numbers (you can't have -3 customers)
- You often have small whole numbers (0, 1, 2, 3, etc.)
- The data often clusters near zero with a "long tail" of occasional higher values

Regular regression methods can give nonsensical results for this kind of data (like predicting -2.5 customers).
Poisson regression specifically handles count data by:

- Always predicting non-negative values
- Understanding the special patterns common in count data
- Properly handling the case when counts are zero

This class lets you configure how the algorithm searches for the best model to fit your count data.

## How It Works

Poisson Regression is particularly suited for analyzing count data where the response variable represents
the number of occurrences of an event within a fixed period or space. Unlike linear regression, which assumes
a normal distribution of errors, Poisson regression assumes the response variable follows a Poisson distribution.
This makes it appropriate for cases where data is discrete, non-negative, and often skewed. The model uses a 
logarithmic link function to ensure predictions are always positive. Poisson regression is widely used in fields
such as epidemiology, insurance (claim counts), ecology (species counts), and marketing (number of purchases).

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type to use when solving the weighted least squares problem. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed in the model fitting process. |
| `Tolerance` | Gets or sets the convergence tolerance threshold for the model fitting process. |

