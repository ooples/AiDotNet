---
title: "PolynomialRegressionOptions<T>"
description: "Configuration options for Polynomial Regression, an extension of linear regression that models the relationship between variables using polynomial functions to capture non-linear relationships in data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Polynomial Regression, an extension of linear regression that models
the relationship between variables using polynomial functions to capture non-linear relationships in data.

## For Beginners

Polynomial Regression is like an upgraded version of regular linear regression that can handle curved relationships.

Imagine you're trying to model this relationship:

- Regular (linear) regression can only draw straight lines
- But many real-world relationships follow curves, not straight lines
- Examples: plant growth over time, diminishing returns on investment, learning curves

What polynomial regression does:

- It adds "power terms" to your equation (squared, cubed, etc.)
- This lets your model create curves instead of just straight lines
- A degree 2 polynomial can make parabolas (U-shapes)
- A degree 3 polynomial can make S-curves
- Higher degrees can create more complex shapes

Think of it like drawing tools:

- Linear regression gives you only a ruler (straight lines)
- Polynomial regression gives you flexible curve tools
- The degree setting controls how flexible those curves can be

This class lets you configure how curved or complex your model can be by setting the polynomial degree.

## How It Works

Polynomial Regression transforms the original features by adding polynomial terms (squared, cubed, etc.)
to the regression equation. This allows the model to fit curved or more complex relationships that cannot
be adequately represented by a straight line. While more flexible than linear regression, polynomial models
with higher degrees can lead to overfitting if not properly regularized or validated. Polynomial regression
is widely used in various fields including economics, social sciences, biology, and engineering when relationships
between variables are suspected to be non-linear. The implementation typically involves creating new features
by raising the original features to various powers, then applying standard linear regression techniques.

## Properties

| Property | Summary |
|:-----|:--------|
| `Degree` | Gets or sets the degree of the polynomial used in the regression model. |

