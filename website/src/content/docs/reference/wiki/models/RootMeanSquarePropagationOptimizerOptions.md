---
title: "RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Root Mean Square Propagation (RMSProp) optimizer, an adaptive learning rate optimization algorithm commonly used in training neural networks."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Root Mean Square Propagation (RMSProp) optimizer, an adaptive learning
rate optimization algorithm commonly used in training neural networks.

## For Beginners

RMSProp is an optimization algorithm that helps neural networks learn more efficiently.

When training a neural network or other machine learning model:

- We need to adjust the model's parameters to minimize errors
- Different parameters may need different adjustment rates
- Some directions in the parameter space may need larger or smaller steps

RMSProp solves these problems by:

- Tracking the recent history of gradients (how parameters should change)
- Automatically adjusting the learning rate for each parameter
- Making larger updates for parameters with small or infrequent gradients
- Making smaller updates for parameters with large or frequent gradients

This adaptive behavior helps the model:

- Learn faster overall
- Avoid getting stuck in poor solutions
- Handle different types of features more effectively

RMSProp is particularly good for:

- Deep neural networks
- Recurrent neural networks
- Problems where different parameters need different learning rates

This class lets you configure the specific behavior of the RMSProp optimizer.

## How It Works

RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to 
address the diminishing learning rates problem of AdaGrad. Proposed by Geoffrey Hinton, RMSProp divides 
the learning rate for each parameter by a running average of the magnitudes of recent gradients for that 
parameter. Unlike AdaGrad, which accumulates all past squared gradients, RMSProp uses an exponentially 
decaying average, which prevents the learning rate from becoming infinitesimally small over time. This 
makes RMSProp particularly well-suited for non-stationary objectives and problems with noisy gradients. 
This class extends GradientBasedOptimizerOptions to provide specific configuration parameters for the 
RMSProp algorithm, including the decay rate for the moving average and a small epsilon value to prevent 
division by zero.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Decay` | Gets or sets the decay rate for the moving average of squared gradients. |
| `Epsilon` | Gets or sets a small constant added to the denominator to improve numerical stability. |

