---
title: "ResNetConfiguration"
description: "Configuration options for ResNet (Residual Network) neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for ResNet (Residual Network) neural network architectures.

## For Beginners

ResNet networks are deep convolutional neural networks that use skip connections
to enable training of very deep architectures. This configuration lets you choose which ResNet variant
to use (ResNet18, ResNet34, ResNet50, ResNet101, or ResNet152), set the number of output classes for
your classification task, and optionally customize the input image dimensions and other parameters.

## How It Works

This configuration class provides all the settings needed to instantiate a ResNet network.
It follows the AiDotNet pattern where users provide minimal configuration and the library
supplies sensible defaults.

**Typical Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResNetConfiguration(ResNetVariant,Int32,Int32,Int32,Int32,Boolean,Boolean)` | Initializes a new instance of the `ResNetConfiguration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` | Gets the base channel counts for each stage. |
| `BlockCounts` | Gets the block counts for each of the 4 stages based on the variant. |
| `Expansion` | Gets the expansion factor for the blocks. |
| `IncludeClassifier` | Gets whether to include the fully connected classifier layer. |
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the height of input images in pixels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `InputWidth` | Gets the width of input images in pixels. |
| `NumClasses` | Gets the number of output classes for classification. |
| `NumConvLayers` | Gets the total number of convolutional layers in the network. |
| `NumWeightLayers` | Gets the total number of weight layers (conv + FC). |
| `TotalInputSize` | Gets the total number of input features (channels * height * width). |
| `UseAutodiff` | Gets or sets whether to use automatic differentiation for backpropagation. |
| `UsesBottleneck` | Gets whether this variant uses BasicBlock (ResNet18/34) or BottleneckBlock (ResNet50/101/152). |
| `Variant` | Gets the ResNet variant to use. |
| `ZeroInitResidual` | Gets whether to use zero-initialization for the last batch normalization in each residual block. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateForCIFAR(ResNetVariant,Int32)` | Creates a configuration optimized for CIFAR-10/CIFAR-100 datasets. |
| `CreateForTesting(Int32)` | Creates a minimal ResNet configuration optimized for fast test execution. |
| `CreateLightweight(Int32)` | Creates a lightweight configuration using ResNet18. |
| `CreateResNet50(Int32)` | Creates a new configuration for ResNet50. |

