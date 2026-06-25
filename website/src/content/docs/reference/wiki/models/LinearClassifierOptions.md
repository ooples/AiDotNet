---
title: "LinearClassifierOptions<T>"
description: "Configuration options for linear classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for linear classifiers.

## For Beginners

Linear classifiers draw a straight line (or hyperplane in
higher dimensions) to separate different classes of data.

Key concepts:

- They learn weights for each feature
- Prediction is: sign(weight * features + bias)
- Training adjusts weights to minimize errors

Linear classifiers are great when:

- You have many features (high-dimensional data)
- Data is approximately linearly separable
- You need fast training and prediction
- You want interpretable models (feature importance = weight magnitude)

## How It Works

Linear classifiers learn a linear decision boundary to separate classes.
They are simple, interpretable, and often very effective.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the regularization strength (alpha). |
| `FitIntercept` | Gets or sets whether to fit an intercept term (bias). |
| `LearningRate` | Gets or sets the learning rate for gradient-based optimization. |
| `Loss` | Gets or sets the loss function type. |
| `MaxIterations` | Gets or sets the maximum number of training iterations. |
| `Penalty` | Gets or sets the penalty type for regularization. |
| `Shuffle` | Gets or sets whether to shuffle training data at each epoch. |
| `Tolerance` | Gets or sets the convergence tolerance. |

