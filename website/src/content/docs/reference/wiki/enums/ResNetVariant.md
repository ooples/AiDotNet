---
title: "ResNetVariant"
description: "Defines the available ResNet (Residual Network) architecture variants."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the available ResNet (Residual Network) architecture variants.

## For Beginners

ResNet networks are named after their total number of weight layers.
For example, ResNet50 has 50 convolutional and fully-connected layers. These networks can be
much deeper than earlier architectures (like VGG) because the skip connections allow gradients
to flow more easily during training, solving the "vanishing gradient" problem.

## How It Works

ResNet architectures are a family of deep convolutional neural networks that introduced skip connections
(residual connections) to enable training of very deep networks. The key innovation is learning residual
functions with reference to the layer inputs, rather than learning unreferenced functions.

**Architecture Types:**

- ResNet18/34 use "BasicBlock" with two 3x3 convolutions
- ResNet50/101/152 use "BottleneckBlock" with 1x1, 3x3, 1x1 convolutions for efficiency

## Fields

| Field | Summary |
|:-----|:--------|
| `ResNet101` | ResNet-101: 101 weight layers using BottleneckBlock. |
| `ResNet152` | ResNet-152: 152 weight layers using BottleneckBlock. |
| `ResNet18` | ResNet-18: 18 weight layers using BasicBlock. |
| `ResNet34` | ResNet-34: 34 weight layers using BasicBlock. |
| `ResNet50` | ResNet-50: 50 weight layers using BottleneckBlock. |

