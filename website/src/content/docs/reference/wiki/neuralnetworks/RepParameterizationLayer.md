---
title: "RepParameterizationLayer<T>"
description: "Represents a reparameterization layer used in variational autoencoders (VAEs) to enable backpropagation through random sampling."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a reparameterization layer used in variational autoencoders (VAEs) to enable backpropagation through random sampling.

## For Beginners

This layer is a special component used in variational autoencoders (VAEs).

Think of the RepParameterizationLayer as a clever randomizer with memory:

- It takes information about a range of possible values (represented by mean and variance)
- It generates random samples from this range
- It remembers how it generated these samples so it can learn during training

For example, in a VAE generating faces:

- Input might represent "average nose size is 5 with variation of ±2"
- This layer randomly picks a specific nose size (like 6.3) based on those statistics
- But it does this in a way that allows the network to learn better statistics

The "reparameterization trick" is what makes this possible - it separates the random sampling
(which can't be directly learned from) from the statistical parameters (which can be learned).

This layer is crucial for variational autoencoders to learn meaningful latent representations
while still incorporating randomness, which helps with generating diverse outputs.

## How It Works

The RepParameterizationLayer implements the reparameterization trick commonly used in variational autoencoders.
It takes an input tensor that contains means and log variances of a latent distribution, samples from this
distribution using the reparameterization trick, and outputs the sampled values. This approach allows
gradients to flow through the random sampling process, which is essential for training VAEs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RepParameterizationLayer` | Initializes a new instance of the `RepParameterizationLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the reparameterization layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for the reparameterization trick. |
| `GetParameters` | Gets all trainable parameters of the reparameterization layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output halves the last dim (mean+logvar split). |
| `ResetState` | Resets the internal state of the reparameterization layer. |
| `UpdateParameters()` | Updates the parameters of the reparameterization layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastEpsilon` | Stores the random noise values used during the sampling process in the forward pass. |
| `_lastLogVar` | Stores the log variance values extracted from the input tensor during the forward pass. |
| `_lastMean` | Stores the mean values extracted from the input tensor during the forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

