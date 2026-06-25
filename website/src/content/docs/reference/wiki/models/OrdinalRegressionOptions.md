---
title: "OrdinalRegressionOptions<T>"
description: "Configuration options for Ordinal Regression (Proportional Odds Model), a classification method for predicting ordered categorical outcomes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Ordinal Regression (Proportional Odds Model), a classification method
for predicting ordered categorical outcomes.

## For Beginners

Ordinal Regression is the right choice when:

- Your categories have a natural order (1 < 2 < 3 < 4 < 5)
- The distances between categories may not be equal
- You want predictions that respect the ordering

Examples:

- Customer satisfaction: Very Unsatisfied, Unsatisfied, Neutral, Satisfied, Very Satisfied
- Movie ratings: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
- Pain levels: None, Mild, Moderate, Severe
- Education level: High School, Some College, Bachelor's, Master's, PhD

The model learns "thresholds" that separate the ordered categories. Each feature can
push predictions up or down the ordinal scale.

## How It Works

Ordinal Regression is used when the target variable has naturally ordered categories, such as
survey responses (strongly disagree to strongly agree), star ratings (1-5 stars), or disease
severity (none, mild, moderate, severe). Unlike regular classification which ignores the order,
ordinal regression models the cumulative probabilities using ordered thresholds.

The model uses the proportional odds (cumulative logit) assumption:
P(Y ≤ k) = sigmoid(α_k - β^T × x)
where α_k are ordered thresholds (cutpoints) and β are the feature coefficients.

## Properties

| Property | Summary |
|:-----|:--------|
| `FitIntercept` | Gets or sets whether to fit an intercept term. |
| `LearningRate` | Gets or sets the learning rate for gradient descent optimization. |
| `LinkFunction` | Gets or sets the link function type for the ordinal regression. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `RegularizationStrength` | Gets or sets the regularization strength (L2 penalty). |
| `Tolerance` | Gets or sets the convergence tolerance for the optimization algorithm. |

