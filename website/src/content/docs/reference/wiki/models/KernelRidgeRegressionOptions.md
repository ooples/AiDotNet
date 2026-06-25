---
title: "KernelRidgeRegressionOptions"
description: "Configuration options for Kernel Ridge Regression, which combines ridge regression with the kernel trick to model non-linear relationships in data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Kernel Ridge Regression, which combines ridge regression with the kernel trick
to model non-linear relationships in data.

## For Beginners

Kernel Ridge Regression is a powerful technique that helps your model
capture complex, non-linear patterns in your data.

Imagine you have data points that can't be fit well with a straight line. For example, the relationship
between a car's speed and its fuel efficiency might be more of a U-shape (very efficient at moderate
speeds, less efficient at very low or very high speeds). Kernel Ridge Regression can help model these
curved relationships.

The "kernel trick" is like giving your model special glasses that let it see your data transformed in
a way that makes complex patterns easier to find. Instead of trying to fit a straight line to a curve,
it transforms the problem so that a straight line in the transformed space corresponds to a curve in
the original space.

The "ridge" part helps prevent overfitting by keeping the model from becoming too complex, similar to
how guardrails keep a car on the road.

This class inherits from NonLinearRegressionOptions, so all the general non-linear regression settings
are also available. The additional settings specific to Kernel Ridge Regression let you fine-tune how
the algorithm balances fitting the data versus keeping the model simple.

## How It Works

Kernel Ridge Regression (KRR) extends standard ridge regression by applying the "kernel trick" to
implicitly map input features into a higher-dimensional space where linear relationships can better
capture complex patterns in the data. This allows the model to learn non-linear relationships while
still maintaining the computational benefits and regularization of ridge regression.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the type of matrix decomposition used to solve the kernel ridge regression equations. |
| `LambdaKRR` | Gets or sets the regularization parameter (lambda) for Kernel Ridge Regression. |

