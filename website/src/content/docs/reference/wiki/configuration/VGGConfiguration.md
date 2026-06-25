---
title: "VGGConfiguration"
description: "Configuration options for VGG neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for VGG neural network architectures.

## For Beginners

VGG networks are deep convolutional neural networks designed for image
classification. This configuration lets you choose which VGG variant to use (VGG11, VGG13,
VGG16, or VGG19), set the number of output classes for your classification task, and optionally
customize the input image dimensions and other parameters.

## How It Works

This configuration class provides all the settings needed to instantiate a VGG network.
It follows the AiDotNet pattern where users provide minimal configuration and the library
supplies sensible defaults.

**Typical Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VGGConfiguration(VGGVariant,Int32,Int32,Int32,Int32,Double,Boolean)` | Initializes a new instance of the `VGGConfiguration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlockConfiguration` | Gets the layer configuration for each VGG block. |
| `DropoutRate` | Gets the dropout rate applied to the fully connected layers. |
| `IncludeClassifier` | Gets whether to include the fully connected classifier layers. |
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the height of input images in pixels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `InputWidth` | Gets the width of input images in pixels. |
| `NumClasses` | Gets the number of output classes for classification. |
| `NumConvLayers` | Gets the number of convolutional layers based on the variant. |
| `NumWeightLayers` | Gets the total number of weight layers (conv + FC). |
| `TotalInputSize` | Gets the total number of input features (channels * height * width). |
| `UseAutodiff` | Gets or sets whether to use automatic differentiation for backpropagation. |
| `UseBatchNormalization` | Gets whether to use batch normalization. |
| `Variant` | Gets the VGG variant to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateForCIFAR(VGGVariant,Int32)` | Creates a configuration optimized for CIFAR-10/CIFAR-100 datasets. |
| `CreateVGG16BN(Int32)` | Creates a new configuration for VGG16 with batch normalization. |

