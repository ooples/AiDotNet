---
title: "MobileNetV2Configuration"
description: "Configuration options for MobileNetV2 neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for MobileNetV2 neural network architectures.

## For Beginners

MobileNetV2 is optimized for mobile devices. The width multiplier
lets you trade accuracy for speed - smaller values give faster but less accurate models.

## How It Works

MobileNetV2 is designed for efficient mobile and edge deployment, using inverted residuals
and linear bottlenecks to achieve high accuracy with low computational cost.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MobileNetV2Configuration(MobileNetV2WidthMultiplier,Int32,Int32,Int32,Int32)` | Initializes a new instance of the `MobileNetV2Configuration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha value (width multiplier as a double). |
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the height of input images in pixels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `InputWidth` | Gets the width of input images in pixels. |
| `NumClasses` | Gets the number of output classes for classification. |
| `WidthMultiplier` | Gets the width multiplier for the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateStandard(Int32)` | Creates a MobileNetV2 configuration with standard width (recommended default). |

