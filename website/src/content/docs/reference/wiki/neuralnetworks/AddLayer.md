---
title: "AddLayer<T>"
description: "A layer that adds multiple input tensors element-wise and optionally applies an activation function."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A layer that adds multiple input tensors element-wise and optionally applies an activation function.

## For Beginners

This layer adds together multiple inputs of the same shape.

Think of this layer as performing element-wise addition:

- If you have two 3×3 matrices, it adds corresponding elements together
- All inputs must have exactly the same dimensions
- After adding, it can optionally apply an activation function

This is commonly used in:

- Residual networks (ResNets) where outputs from earlier layers are added to later layers
- Skip connections that help information flow more directly through deep networks
- Any situation where you want to combine information from multiple sources

For example, if you have two feature maps from different parts of a network,
this layer lets you combine them by adding their values together.

## How It Works

The AddLayer combines multiple tensors of identical shape by adding their values element-wise. This is useful for 
implementing residual connections, skip connections, or any architecture that requires combining information from 
multiple sources. After adding the inputs, an optional activation function can be applied to the result.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Indicates whether this layer has trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeActivationBackwardGpu(DirectGpuTensorEngine,Tensor<>,Tensor<>,FusedActivationType)` | Computes the activation backward gradient on GPU. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | This method is not supported for AddLayer, which requires multiple inputs. |
| `Forward(Tensor<>[])` | Processes multiple input tensors by adding them element-wise and optionally applying an activation function. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Gets all trainable parameters of this layer as a flat vector. |
| `ResetState` | Clears the layer's memory of previous inputs and outputs. |
| `UpdateParameters()` | Updates the layer's internal parameters during training. |
| `ValidateInputShapes(Int32[][])` | Validates that there are at least two input shapes and that all shapes are identical. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports for this multi-input layer. |
| `_lastInputs` | Stores the input tensors from the most recent forward pass for use in the backward pass. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass for use in the backward pass. |

