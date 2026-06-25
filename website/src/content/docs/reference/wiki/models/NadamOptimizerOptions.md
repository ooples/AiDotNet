---
title: "NadamOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Nadam optimizer, which combines Nesterov momentum with Adam's adaptive learning rates for efficient training of neural networks and other gradient-based models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Nadam optimizer, which combines Nesterov momentum with Adam's
adaptive learning rates for efficient training of neural networks and other gradient-based models.

## For Beginners

Nadam is an advanced optimization algorithm that helps neural networks
and other machine learning models learn more efficiently.

Imagine you're trying to navigate to the lowest point in a hilly landscape while blindfolded:

- Standard gradient descent is like taking steps directly downhill from where you're standing
- Adam adds adaptive step sizes (taking bigger steps in flat areas, smaller steps in steep areas)
- Nadam goes a step further by trying to predict where you'll be after your next step and looking

at the downhill direction from that predicted position

This combination of techniques helps the algorithm:

- Learn faster than simpler methods
- Avoid getting stuck in small dips that aren't the true lowest point
- Adapt to different parts of the learning process with appropriate step sizes

This class lets you fine-tune how Nadam works: how quickly it learns, how much it relies on past
information, and how it adapts its learning rate during training.

## How It Works

Nadam (Nesterov-accelerated Adaptive Moment Estimation) is an optimization algorithm that extends
Adam by incorporating Nesterov momentum. Like Adam, it maintains adaptive learning rates for each
parameter based on estimates of first and second moments of the gradients. Additionally, it applies
the Nesterov acceleration technique, which evaluates the gradient at a "look-ahead" position rather
than the current position. This combination often leads to faster convergence than standard Adam,
particularly for problems with complex loss landscapes or sparse gradients.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates (momentum). |
| `Beta2` | Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rates). |
| `Epsilon` | Gets or sets a small constant added to the denominator to improve numerical stability. |
| `InitialLearningRate` | Gets or sets the initial learning rate that controls the step size in parameter updates. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate is decreased when the loss is getting worse. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate is increased when the loss is improving. |
| `MaxLearningRate` | Gets or sets the maximum allowed value for the learning rate. |
| `MinLearningRate` | Gets or sets the minimum allowed value for the learning rate. |

