---
title: "ResNet<T>"
description: "ResNet backbone network for feature extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Backbones`

ResNet backbone network for feature extraction.

## For Beginners

ResNet (Residual Network) is a foundational architecture
that introduced skip connections to enable training of very deep networks. It's widely
used as a backbone for detection models like Faster R-CNN.

## How It Works

Key features:

- Residual blocks with skip connections prevent gradient vanishing
- Multiple variants: ResNet-18, 34, 50, 101, 152
- Bottleneck blocks (3 convolutions) for deeper networks

Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResNet(ResNetVariant,Int32,IActivationFunction<>)` | Creates a new ResNet backbone. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Sum across the stem conv plus every residual stage. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeepCopy` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activation` | Activation between stages. |

