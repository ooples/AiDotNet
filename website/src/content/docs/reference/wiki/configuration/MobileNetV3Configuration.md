---
title: "MobileNetV3Configuration"
description: "Configuration options for MobileNetV3 neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for MobileNetV3 neural network architectures.

## For Beginners

MobileNetV3 comes in Large (more accurate) and Small (faster)
variants. Choose based on whether you prioritize accuracy or speed.

## How It Works

MobileNetV3 builds on MobileNetV2 with additional optimizations including
squeeze-and-excitation blocks and hard-swish activation for improved accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MobileNetV3Configuration(MobileNetV3Variant,Int32,MobileNetV3WidthMultiplier,Int32,Int32,Int32)` | Initializes a new instance of the `MobileNetV3Configuration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha value (width multiplier as a double). |
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the height of input images in pixels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `InputWidth` | Gets the width of input images in pixels. |
| `NumClasses` | Gets the number of output classes for classification. |
| `Variant` | Gets the MobileNetV3 variant to use. |
| `WidthMultiplier` | Gets the width multiplier for the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateLarge(Int32)` | Creates a MobileNetV3-Large configuration (recommended default). |
| `CreateSmall(Int32)` | Creates a MobileNetV3-Small configuration for low-latency applications. |

