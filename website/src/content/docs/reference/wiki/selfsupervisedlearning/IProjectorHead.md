---
title: "IProjectorHead<T>"
description: "Defines the contract for projection heads used in self-supervised learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.SelfSupervisedLearning`

Defines the contract for projection heads used in self-supervised learning.

## For Beginners

A projection head is a small neural network that transforms
encoder outputs into a space optimized for the SSL loss. After pretraining, the projection
head is typically discarded and only the encoder is used for downstream tasks.

## How It Works

**Why use a projection head?**

**Common architectures:**

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` | Gets the hidden dimension (for MLP projectors). |
| `InputDimension` | Gets the input dimension expected by this projector. |
| `OutputDimension` | Gets the output dimension produced by this projector. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Clears accumulated gradients. |
| `GetParameterGradients` | Gets the gradients computed during the last backward pass. |
| `GetParameters` | Gets all trainable parameters of the projector. |
| `Project(Tensor<>)` | Projects encoder output to the SSL embedding space. |
| `Reset` | Resets the projector state (clears any internal buffers). |
| `SetParameters(Vector<>)` | Sets the parameters of the projector. |
| `SetTrainingMode(Boolean)` | Sets training or evaluation mode. |

