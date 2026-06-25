---
title: "AdaMaxOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the AdaMax optimization algorithm, a variant of Adam that uses the infinity norm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AdaMax optimization algorithm, a variant of Adam that uses the infinity norm.

## For Beginners

AdaMax is like a specialized version of a popular learning algorithm (Adam) that's
particularly good at handling situations where most values are zero with occasional large values.
Think of it as a specialized tool that works better than general-purpose tools for certain specific tasks.

## How It Works

AdaMax is a variant of the Adam optimizer that uses the infinity norm instead of the L2 norm in the update rule.
This can make it more stable for certain types of problems, especially those with sparse gradients.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates. |
| `Beta2` | Gets or sets the exponential decay rate for the infinity norm of past gradients. |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `InitialLearningRate` | Gets or sets the learning rate, which controls how quickly the model adapts to the problem. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate decreases when performance worsens. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate increases when performance improves. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate, overriding the base class value with a value optimized for AdaMax. |
| `MinLearningRate` | Gets or sets the minimum allowed learning rate, overriding the base class value with a value optimized for AdaMax. |

