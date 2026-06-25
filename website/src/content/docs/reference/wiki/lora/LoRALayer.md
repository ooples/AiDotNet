---
title: "LoRALayer<T>"
description: "Implements Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning of neural networks."
section: "API Reference"
---

`Layers` · `AiDotNet.LoRA`

Implements Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning of neural networks.

## For Beginners

LoRA is a technique that makes it much cheaper to adapt large neural networks
to new tasks. Instead of updating all the weights in a layer (which can be millions of parameters),
LoRA adds two small matrices that work together to approximate the needed changes.

Think of it like this:

- Traditional fine-tuning: Adjusting every single knob on a massive control panel
- LoRA: Using just a few master controls that influence many knobs at once

The key insight is that the changes needed for fine-tuning often lie in a "low-rank" space,
meaning we don't need full freedom to adjust every parameter independently.

Key parameters:

- Rank (r): Controls how many "master controls" you have. Higher rank = more flexibility but more parameters
- Alpha: A scaling factor that controls how much influence the LoRA adaptation has

For example, adapting a layer with 1000x1000 weights (1M parameters) using LoRA with rank=8 only
requires 8x1000 + 8x1000 = 16,000 parameters (98.4% reduction!).

## How It Works

LoRA works by decomposing weight updates into two low-rank matrices A and B, where the actual update
is computed as B * A. This dramatically reduces the number of trainable parameters compared to
fine-tuning all weights directly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoRALayer(Int32,Int32,Int32,Double,IActivationFunction<>)` | Initializes a new LoRA layer with the specified dimensions and hyperparameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha scaling factor. |
| `ParameterCount` | Gets the total number of trainable parameters (elements in A and B matrices). |
| `Rank` | Gets the rank of this LoRA layer. |
| `Scaling` | Gets the computed scaling factor (alpha / rank). |
| `SupportsTraining` | Gets whether this layer supports training (always true for LoRA). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through the LoRA layer. |
| `GetMatrixA` | Gets matrix A (for inspection or advanced use cases). |
| `GetMatrixB` | Gets matrix B (for inspection or advanced use cases). |
| `GetParameters` | Gets the current parameters as a vector. |
| `MergeWeights` | Merges the LoRA weights into a dense weight matrix that can be added to a base layer. |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateMatricesFromParameters` | Updates the matrices from the parameter vector. |
| `UpdateParameterGradients` | Updates the parameter gradients vector from the matrix gradients. |
| `UpdateParameters()` | Updates the layer's parameters using the specified learning rate. |
| `UpdateParametersFromMatrices` | Updates the parameter vector from the current matrix values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | Scaling factor for the LoRA contribution. |
| `_lastInput` | Stored input from the forward pass, needed for gradient computation. |
| `_lastPreActivation` | Stored pre-activation output from the forward pass, needed for activation derivative computation. |
| `_loraA` | Low-rank matrix A with dimensions (inputSize × rank). |
| `_loraATensor` | Gradients for matrix A computed during backpropagation. |
| `_loraB` | Low-rank matrix B with dimensions (rank × outputSize). |
| `_loraBGradient` | Gradients for matrix B computed during backpropagation. |
| `_rank` | The rank of the low-rank decomposition. |
| `_scaling` | Computed scaling factor (alpha / rank) used during forward pass. |

