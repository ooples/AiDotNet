---
title: "MaskingLayer<T>"
description: "Represents a layer that masks specified values in the input tensor, typically used to ignore padding in sequential data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that masks specified values in the input tensor, typically used to ignore padding in sequential data.

## For Beginners

This layer helps the network ignore certain parts of your data.

Think of it like a highlighter that marks which parts of your data are important:

- Any value matching the "mask value" (usually 0) gets ignored
- All other values pass through unchanged
- This is especially useful for sequences of different lengths

For example, if you have sentences of different lengths:

- Short sentences might be padded with zeros to match longer ones
- The masking layer tells the network to ignore those zeros
- This helps the network focus only on the real data

Without masking, the network would try to learn patterns from the padding values,
which would confuse the learning process.

## How It Works

The MaskingLayer is used to skip certain time steps in sequential data by masking out specific values. 
During the forward pass, time steps with values equal to the mask value are multiplied by zero, effectively 
removing them from consideration by subsequent layers. This is particularly useful for handling variable-length 
sequences where padding is used to make all sequences the same length.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskingLayer(Double)` | Initializes a new instance of the `MaskingLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMask(Tensor<>,Tensor<>)` | Applies a binary mask to an input tensor through element-wise multiplication. |
| `CreateMask(Tensor<>)` | Creates a binary mask from the input tensor based on the mask value. |
| `Forward(Tensor<>)` | Performs the forward pass of the masking layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-resident forward pass of the masking layer. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output equals input shape (passthrough). |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastMask` | The mask tensor from the last forward pass. |
| `_lastMaskGpu` | The GPU mask tensor from the last GPU forward pass (for backward pass caching). |
| `_maskValue` | The value to be masked out in the input tensor. |

