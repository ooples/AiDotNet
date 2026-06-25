---
title: "GradientDescentOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Gradient Descent optimizer, which is a fundamental algorithm for finding the minimum of a function by iteratively moving in the direction of steepest descent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Gradient Descent optimizer, which is a fundamental algorithm for
finding the minimum of a function by iteratively moving in the direction of steepest descent.

## For Beginners

Think of Gradient Descent like finding the lowest point in a valley by
always walking downhill. Imagine you're standing on a hilly landscape and want to reach the lowest
point. You look around, figure out which direction is most steeply downhill, take a step in that
direction, and repeat until you can't go any lower.

In machine learning, the "landscape" is the error or loss function (how wrong your model's predictions
are), and the "lowest point" represents the best possible model parameters. Gradient descent helps your
model learn by repeatedly adjusting its parameters to reduce prediction errors.

This is the most basic form of optimization used in many machine learning algorithms, including neural
networks, linear regression, and logistic regression. The options in this class let you control how
quickly the algorithm moves downhill and how it avoids certain pitfalls during the optimization process.

## How It Works

Gradient Descent is one of the most widely used optimization algorithms in machine learning. It works
by calculating the gradient (slope) of a loss function with respect to the model parameters, then
updating those parameters in the opposite direction of the gradient to minimize the loss. This class
inherits from GradientBasedOptimizerOptions, so all general gradient-based optimization settings are
also available.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientDescentOptimizerOptions` | Initializes a new instance of the `GradientDescentOptimizerOptions` class with default settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `RegularizationOptions` | Gets or sets the regularization options to control overfitting during optimization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefaultRegularizationOptions` | Creates default regularization options specifically tuned for gradient descent optimization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_regularizationOptions` | The regularization options to control overfitting during gradient descent optimization. |

