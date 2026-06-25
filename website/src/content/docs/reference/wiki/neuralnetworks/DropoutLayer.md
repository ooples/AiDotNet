---
title: "DropoutLayer<T>"
description: "Implements a dropout layer for neural networks to prevent overfitting."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a dropout layer for neural networks to prevent overfitting.

## For Beginners

Dropout is like randomly turning off some brain cells during training to make the network more robust.

Imagine a team that always practices together:

- They might develop specific patterns that only work with familiar teammates
- If some players are absent, the team struggles

Dropout forces the network to work even when some neurons are missing:

- During training, random neurons are turned off (set to zero)
- This prevents any single neuron from becoming too important
- The network learns multiple ways to solve the same problem
- It's like practicing with different team combinations each time

During actual use (inference), all neurons are active, but their outputs are slightly reduced
to compensate for having more active neurons than during training.

This technique significantly reduces overfitting, which is when a network gets too specialized
to its training data and performs poorly on new data.

## How It Works

Dropout is a regularization technique that randomly deactivates a fraction of neurons during
training, which helps prevent neural networks from overfitting. Overfitting occurs when a model
learns patterns that are specific to the training data but don't generalize well to new data.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvanceSeedCounter` | Advances the internal forward-call counter atomically and returns the PRE-increment value (i.e., the same semantics as a post-fix `_seedCounter++`). |
| `ConvertToOnnx(OnnxGraphBuilder,OnnxLayerInputs)` | Dropout in inference mode is a no-op. |
| `DeriveSeed32(Int32,UInt64)` | Shared seed-derivation helper for the CPU Forward path. |
| `DeriveSeed64(Int32,UInt64)` | Shared seed-derivation helper for the GPU Forward path. |
| `Forward(Tensor<>)` | Performs the forward pass of the dropout layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU with full training support. |
| `GetMetadata` | Returns layer-specific metadata required for cloning/serialization. |
| `GetParameters` | Gets the trainable parameters of the layer. |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dropoutMask` | The binary mask that indicates which neurons were kept active during the last forward pass. |
| `_dropoutRate` | The probability of dropping out (deactivating) a neuron during training. |
| `_gpuDropoutMask` | The GPU-resident dropout mask from the last GPU forward pass. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_scale` | The scaling factor applied to active neurons during training. |
| `_seedCounter` | Counter for generating unique random seeds per forward pass. |

