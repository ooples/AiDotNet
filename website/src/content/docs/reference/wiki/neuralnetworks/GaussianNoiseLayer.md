---
title: "GaussianNoiseLayer<T>"
description: "A neural network layer that adds random Gaussian noise to inputs during training."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A neural network layer that adds random Gaussian noise to inputs during training.

## For Beginners

This layer adds random "static" to your data during training to make the network more robust.

Think of it like training an athlete in challenging conditions:

- Training in rain and wind makes athletes perform better even in good weather
- Training with noise makes neural networks perform better on clean data

For example, in image recognition:

- During training: The layer slightly changes pixel values randomly
- This forces the network to focus on important patterns, not tiny details
- During testing/prediction: No noise is added, giving clean results

Gaussian noise is particularly useful because it follows the same distribution
as many natural variations in real-world data.

## How It Works

Gaussian noise layers help prevent overfitting by adding random noise to the input data.
This forces the network to learn more robust features that can withstand small variations.
The noise follows a Gaussian (normal) distribution with a specified mean and standard deviation.
During inference (testing/prediction), no noise is added to preserve predictable outputs.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass by adding Gaussian noise to the input during training. |
| `ForwardGpu(Tensor<>[])` |  |
| `GenerateNoise(Int32[])` | Generates a tensor of random Gaussian noise with the specified shape. |
| `GetParameters` | Gets the trainable parameters of the layer. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output equals input (passthrough). |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastNoise` | The noise tensor from the last forward pass, saved for potential use in backpropagation. |
| `_mean` | The mean (average value) of the Gaussian noise distribution. |
| `_standardDeviation` | The standard deviation of the Gaussian noise distribution. |

