---
title: "DenseNetConfiguration"
description: "Configuration options for DenseNet neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for DenseNet neural network architectures.

## For Beginners

DenseNet is designed to maximize information flow by connecting each
layer directly to all subsequent layers. This configuration lets you choose which variant
to use and customize parameters like growth rate and compression factor.

## How It Works

DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer
in a feed-forward fashion, enabling strong gradient flow and feature reuse.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenseNetConfiguration(DenseNetVariant,Int32,Int32,Int32,Int32,Int32,Double,Int32[])` | Initializes a new instance of the `DenseNetConfiguration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionFactor` | Gets the compression factor for transition layers. |
| `CustomBlockLayers` | Gets the custom block layers configuration (only used when Variant is Custom). |
| `GrowthRate` | Gets the growth rate (k in the paper). |
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the height of input images in pixels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `InputWidth` | Gets the width of input images in pixels. |
| `NumClasses` | Gets the number of output classes for classification. |
| `Variant` | Gets the DenseNet variant to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDenseNet121(Int32)` | Creates a DenseNet-121 configuration (recommended default). |
| `CreateForTesting(Int32)` | Creates a minimal DenseNet configuration optimized for fast test execution. |
| `GetBlockLayers` | Gets the number of layers per dense block for this variant. |
| `GetExpectedLayerCount` | Gets the expected total layer count for this configuration without constructing the network. |

