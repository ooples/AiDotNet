---
title: "MaskHead<T>"
description: "Mask prediction head for instance segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.InstanceSegmentation`

Mask prediction head for instance segmentation.

## For Beginners

The mask head takes RoI-pooled features and predicts
a binary segmentation mask for each class. It typically uses a series of
convolutional layers followed by a transposed convolution for upsampling.

## How It Works

Key features:

- Multiple convolutional layers for feature processing
- Upsampling via transposed convolution
- Per-class mask prediction
- Configurable mask resolution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskHead(Int32,Int32,Int32)` | Creates a new mask head. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Forward pass to predict masks from RoI features. |
| `GetParameterCount` | Gets the total parameter count. |
| `PredictMask(Tensor<>,Int32)` | Predicts mask for a single RoI and class. |
| `ReadParameters(BinaryReader)` | Reads parameters from binary reader. |
| `WriteParameters(BinaryWriter)` | Writes parameters to binary writer. |

