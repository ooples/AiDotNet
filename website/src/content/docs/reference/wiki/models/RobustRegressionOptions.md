---
title: "RobustRegressionOptions<T>"
description: "Configuration options for robust regression models, which are designed to be less sensitive to outliers and violations of standard regression assumptions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for robust regression models, which are designed to be less sensitive
to outliers and violations of standard regression assumptions.

## For Beginners

Robust regression helps your model handle outliers and unusual data points.

Standard regression can be easily thrown off by outliers (extreme values):

- Imagine predicting house prices where most homes are $200,000-$400,000
- But there's one luxury mansion worth $10 million in your data
- Standard regression might shift significantly to accommodate this outlier

Robust regression solves this problem by:

- Giving less weight to data points that are far from the pattern
- Focusing more on the typical cases that represent the true relationship
- Producing more reliable predictions when your data has unusual values

Think of it like taking a poll:

- Standard regression counts everyone's vote equally
- Robust regression gives more consideration to mainstream opinions and less to extreme views

This class lets you configure exactly how the robust regression algorithm handles outliers
and unusual data points.

## How It Works

Robust regression methods provide an alternative to standard least squares regression when data contains 
outliers or exhibits heteroscedasticity (non-constant variance in errors). Unlike ordinary least squares 
regression, which can be heavily influenced by outliers, robust regression methods use specialized techniques 
to reduce the impact of outlying observations. This class extends the standard RegressionOptions class to 
include additional parameters specific to robust regression algorithms, such as the tuning constant, 
maximum iterations for iterative reweighting procedures, convergence tolerance, weight function selection, 
and an optional initial regression model for initialization. These options allow fine-tuning of the robust 
regression algorithm to best handle the specific characteristics of the dataset being analyzed.

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialRegression` | Gets or sets the initial regression model used to start the iterative procedure. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the iterative reweighting procedure. |
| `Tolerance` | Gets or sets the convergence tolerance for the iterative reweighting procedure. |
| `TuningConstant` | Gets or sets the tuning constant that controls the sensitivity to outliers. |
| `WeightFunction` | Gets or sets the weight function used to reduce the influence of outliers. |

