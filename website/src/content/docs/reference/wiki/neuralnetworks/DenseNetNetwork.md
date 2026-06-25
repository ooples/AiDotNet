---
title: "DenseNetNetwork<T>"
description: "Implements the DenseNet (Densely Connected Convolutional Network) architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the DenseNet (Densely Connected Convolutional Network) architecture.

## For Beginners

DenseNet is designed to maximize information flow
through the network by connecting each layer directly to all subsequent layers.

Key innovations:

- Dense Connectivity: Each layer receives features from ALL previous layers
- Feature Reuse: Reduces redundant feature learning, fewer parameters
- Strong Gradient Flow: Direct connections help train very deep networks
- Compact Models: Can achieve similar accuracy with fewer parameters than ResNet

The "growth rate" (k) determines how many new feature maps each layer adds.
Typical values are 12, 24, or 32. Higher values increase capacity but also cost.

## How It Works

DenseNet (Huang et al., 2017) connects each layer to every other layer in a
feed-forward fashion. This creates strong gradient flow and feature reuse,
enabling very deep networks with fewer parameters.

Architecture overview:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenseNetNetwork` | Initializes a new instance with default settings. |
| `DenseNetNetwork(NeuralNetworkArchitecture<>,DenseNetConfiguration,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,DenseNetOptions)` | Initializes a new instance of the `DenseNetNetwork` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GrowthRate` | Gets the growth rate (k). |
| `NumClasses` | Gets the number of output classes. |
| `Variant` | Gets the DenseNet variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DenseNet121(Int32,Int32)` | Creates a DenseNet-121 network. |
| `DenseNet169(Int32,Int32)` | Creates a DenseNet-169 network. |
| `DenseNet201(Int32,Int32)` | Creates a DenseNet-201 network. |
| `DenseNet264(Int32,Int32)` | Creates a DenseNet-264 network. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ForTesting(Int32,Int32)` | Creates a minimal DenseNet network optimized for fast test execution. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetDenseNetDefaultLoss(NeuralNetworkTaskType)` | Returns the appropriate loss function for DenseNet. |
| `GetLayer(Int32)` | Gets the layer at the specified index. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

