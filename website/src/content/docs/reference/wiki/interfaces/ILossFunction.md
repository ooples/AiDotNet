---
title: "ILossFunction<T>"
description: "Interface for loss functions used in neural networks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for loss functions used in neural networks.

## For Beginners

Loss functions measure how far the predictions of a neural network are from the expected outputs.
They provide a signal that helps the network learn by adjusting its weights to minimize this "loss" value.

Think of a loss function as a score that tells you how well or poorly your neural network is performing.
A higher loss value means worse performance, while a lower loss value indicates better performance.

Different types of problems require different loss functions. For example:

- Mean Squared Error is often used for regression problems (predicting numeric values)
- Cross Entropy is commonly used for classification problems (categorizing inputs)

The derivative of a loss function is equally important, as it tells the network which direction to adjust
its weights during training to reduce the loss.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` | Calculates the derivative (gradient) of the loss function. |
| `CalculateLoss(Vector<>,Vector<>)` | Calculates the loss between predicted and actual values. |
| `CalculateLossAndGradientGpu(Tensor<>,Tensor<>)` | Calculates both loss and gradient on GPU in a single pass. |

