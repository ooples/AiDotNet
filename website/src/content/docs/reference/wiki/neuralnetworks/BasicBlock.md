---
title: "BasicBlock<T>"
description: "Implements the BasicBlock used in ResNet18 and ResNet34 architectures."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements the BasicBlock used in ResNet18 and ResNet34 architectures.

## For Beginners

The BasicBlock is like a "learning module" with a shortcut.

The key insight is:

- The two conv layers learn to predict what needs to be ADDED to the input (the "residual")
- The skip connection adds the original input back to this learned residual
- This makes it easier to train very deep networks because gradients can flow directly through the skip connection

When the input and output have different dimensions (due to stride or channel changes),
a downsample layer (1x1 conv + BN) is used to match the dimensions before adding.

## How It Works

The BasicBlock contains two 3x3 convolutional layers with batch normalization and ReLU activation.
A skip connection adds the input directly to the output, enabling gradient flow through very deep networks.

**Architecture:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicBlock(Int32,Int32,Boolean)` | Initializes a new instance of the `BasicBlock` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer has a GPU implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through the BasicBlock. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, keeping data GPU-resident. |
| `GetParameters` | Gets all trainable parameters. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the block. |
| `UpdateParameters()` | Updates the parameters of all internal layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Expansion` | The expansion factor for BasicBlock. |

