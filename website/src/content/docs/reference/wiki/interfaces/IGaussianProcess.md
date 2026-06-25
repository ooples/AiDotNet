---
title: "IGaussianProcess<T>"
description: "Defines an interface for Gaussian Process regression, a powerful probabilistic machine learning technique."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for Gaussian Process regression, a powerful probabilistic machine learning technique.

## How It Works

**For Beginners:** A Gaussian Process is a flexible machine learning approach that not only makes predictions 
but also tells you how confident it is about each prediction.

Imagine you're trying to predict house prices in different neighborhoods:

- Traditional models might just say "this house costs $300,000"
- A Gaussian Process would say "this house costs about $300,000, and I'm very confident 

the price is between $290,000 and $310,000"

This is especially useful when:

- You have limited data
- You need to know how certain the model is about its predictions
- You want to make decisions that account for uncertainty

Unlike many other machine learning methods, Gaussian Processes:

- Don't assume a specific form for the relationship between inputs and outputs
- Automatically adapt to the complexity of your data
- Provide uncertainty estimates for each prediction

The "Gaussian" part refers to the normal distribution (bell curve) used to represent uncertainty.
The "Process" part means it works with functions rather than simple values.

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Trains the Gaussian Process model on the provided data. |
| `Predict(Vector<>)` | Predicts the mean value and variance (uncertainty) for a new input point. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used by the Gaussian Process. |

