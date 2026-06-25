---
title: "IGaussianProcessClassifier<T>"
description: "Defines an interface for Gaussian Process classification, a probabilistic approach to classification that provides uncertainty estimates along with class predictions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for Gaussian Process classification, a probabilistic approach to classification
that provides uncertainty estimates along with class predictions.

## For Beginners

Gaussian Process Classification (GPC) is a powerful method that combines
the flexibility of Gaussian Processes with classification tasks.

Unlike regular classifiers that just say "this is class A," a GP classifier tells you:

- "This is probably class A (90% confident)"
- "This might be class A or B (60%/40% split)"

This is incredibly useful when:

- You need to know how confident the model is in its predictions
- You want to identify ambiguous cases that might need human review
- Your data has complex, non-linear patterns
- You have limited training data but need reliable predictions

The GP classifier works by:

1. Learning a latent (hidden) function over your input space
2. Passing this function through a link function (like sigmoid) to get probabilities
3. Using approximation techniques (like Laplace) to handle the non-Gaussian likelihood

Key differences from GP regression:

- Regression predicts continuous values; classification predicts discrete labels
- Classification requires a link function to convert latent values to probabilities
- The posterior is not analytically tractable, requiring approximation methods

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of classes learned during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Trains the Gaussian Process classifier on labeled training data. |
| `GetLogMarginalLikelihood` | Gets the log marginal likelihood of the model, useful for hyperparameter optimization. |
| `Predict(Vector<>)` | Predicts the class probabilities for a new input point. |
| `PredictProbabilities(Matrix<>)` | Predicts class probabilities for multiple input points. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used by the classifier. |

