---
title: "AdaDeltaOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the AdaDelta optimization algorithm, which is an extension of AdaGrad that adapts learning rates based on a moving window of gradient updates."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AdaDelta optimization algorithm, which is an extension of AdaGrad that adapts learning rates based on a moving window of gradient updates.

## For Beginners

AdaDelta is like a smart learning system that automatically adjusts how quickly it learns based on past experience.
Instead of using a fixed learning speed, it slows down for parameters that change a lot and speeds up for those that change little.
This helps the model learn more efficiently without requiring manual tuning of the learning rate.

## How It Works

AdaDelta is an optimization algorithm that dynamically adapts the learning rate for each parameter based on historical gradient information.
It addresses some limitations of earlier algorithms by using a moving average of squared gradients.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the AdaDelta optimizer, overriding the base class value. |
| `MaxRho` | Gets or sets the maximum allowed value for Rho. |
| `MinRho` | Gets or sets the minimum allowed value for Rho. |
| `Rho` | Gets or sets the decay rate for the moving average of squared gradients. |
| `RhoDecreaseFactor` | Gets or sets the factor by which Rho decreases when performance worsens. |
| `RhoIncreaseFactor` | Gets or sets the factor by which Rho increases when performance improves. |
| `UseAdaptiveRho` | Gets or sets whether to automatically adjust the Rho parameter during training. |

