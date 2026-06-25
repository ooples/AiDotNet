---
title: "VAEResBlock<T>"
description: "Residual block for VAE encoder/decoder with GroupNorm and skip connections."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Residual block for VAE encoder/decoder with GroupNorm and skip connections.

## For Beginners

A residual block helps the network learn more effectively.

Think of it like taking notes during a lecture:

- The main path (two convolutions) learns new features
- The skip connection preserves the original information
- Adding them together means you learn the "difference" or "improvement"

The GroupNorm helps stabilize training by normalizing activations within groups
of channels, which works well even with small batch sizes commonly used in
image generation tasks.

Structure:
```
input ─────────────────────────────────┐
│ │
├─→ GroupNorm → SiLU → Conv3x3 ─→ h │ (skip connection)
│ │
│ ↓ │
│ │
│ GroupNorm → SiLU → Conv3x3 ─→ h │
│ │
│ ↓ ↓
│ [1x1 Conv if channels differ]
│ ↓ ↓
└────────────────→ (+) ←─────────────┘
│
output
```

## How It Works

This implements a proper VAE residual block following the Stable Diffusion VAE architecture:

- GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv
- Skip connection with optional 1x1 convolution when input/output channels differ

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VAEResBlock(Int32,Int32,Int32,Int32)` | Initializes a new instance of the VAEResBlock class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels. |
| `NumGroups` | Gets the number of groups for GroupNorm. |
| `OutputChannels` | Gets the number of output channels. |
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySiLU(Tensor<>)` | Applies SiLU activation to a tensor. |
| `ApplySiLUDerivative(Tensor<>,Tensor<>)` | Computes the SiLU derivative for a tensor. |
| `Deserialize(BinaryReader)` | Loads the block's state from a binary reader. |
| `Forward(Tensor<>)` | Performs the forward pass through the residual block. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `ResetState` | Resets the internal state of the block. |
| `Serialize(BinaryWriter)` | Saves the block's state to a binary writer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `UpdateParameters()` | Updates all learnable parameters using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_conv1` | First convolution layer. |
| `_conv2` | Second convolution layer. |
| `_inChannels` | Number of input channels. |
| `_lastInput` | Cached input from forward pass for backward. |
| `_norm1` | First GroupNorm layer. |
| `_norm1Output` | Cached intermediate values for backward pass. |
| `_norm2` | Second GroupNorm layer. |
| `_numGroups` | Number of groups for GroupNorm. |
| `_outChannels` | Number of output channels. |
| `_silu` | SiLU activation function. |
| `_skipConv` | Optional 1x1 convolution for skip connection when channels differ. |

