---
title: "AMSGradOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the AMSGrad optimization algorithm, which is an improved variant of the Adam optimizer that addresses potential convergence issues by maintaining the maximum of past squared gradients."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AMSGrad optimization algorithm, which is an improved variant of the Adam optimizer
that addresses potential convergence issues by maintaining the maximum of past squared gradients.

## For Beginners

AMSGrad is like a smart running coach that adjusts your training pace based on your
past performance. It remembers how difficult different parts of your training have been and adjusts accordingly,
making sure you don't slow down too much on challenging sections. This helps your AI model learn more efficiently
by giving more attention to important patterns and less to noise in the data. Unlike some other methods, AMSGrad
ensures that your learning progress never goes backward, which helps it reach better solutions.

## How It Works

AMSGrad is an adaptive learning rate optimization algorithm that combines the benefits of AdaGrad and RMSProp
while ensuring convergence by using a non-decreasing learning rate adjustment. It's particularly effective for
deep learning models and non-convex optimization problems.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates (momentum). |
| `Beta2` | Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rates). |
| `Epsilon` | Gets or sets a small constant added to denominators to improve numerical stability. |
| `InitialLearningRate` | Gets or sets the initial step size used for parameter updates during optimization. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which to decrease the learning rate when the loss is increasing or oscillating. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which to increase the learning rate when the loss is consistently decreasing. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate during adaptive adjustments. |
| `MinLearningRate` | Gets or sets the minimum allowed learning rate during adaptive adjustments. |

