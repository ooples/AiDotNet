---
title: "ResNetNetwork<T>"
description: "Represents a ResNet (Residual Network) neural network architecture for image classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a ResNet (Residual Network) neural network architecture for image classification.

## For Beginners

ResNet networks revolutionized deep learning by solving the "vanishing gradient"
problem that made very deep networks hard to train. Key benefits include:

- Can train networks with 100+ layers (compared to ~20 layers for earlier architectures)
- Skip connections allow gradients to flow more easily during training
- Each block learns the "residual" (difference) rather than the complete transformation
- Winner of ImageNet 2015 competition with top-5 error of 3.57%

## How It Works

ResNet networks are deep convolutional neural networks that introduced skip connections (residual
connections) to enable training of very deep networks. They learn residual functions with reference
to the layer inputs, rather than learning unreferenced functions directly.

**Architecture Variants:**

- **ResNet18/34:** Use BasicBlock (2 conv layers per block)
- **ResNet50/101/152:** Use BottleneckBlock (1x1-3x3-1x1 conv pattern) for efficiency

**Typical Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResNetNetwork` | Initializes a new instance with default settings. |
| `ResNetNetwork(NeuralNetworkArchitecture<>,ResNetConfiguration,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,ResNetOptions)` | Initializes a new instance of the ResNetNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes for classification. |
| `UsesBottleneck` | Gets whether this variant uses bottleneck blocks. |
| `Variant` | Gets the ResNet variant being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyParameterUpdates` | Applies parameter updates using the optimizer. |
| `CalculateOutputGradient(Tensor<>,Tensor<>)` | Calculates the gradient of the loss with respect to the network's output. |
| `CreateNewInstance` | Creates a new instance of the ResNet network model. |
| `CreateResNetLayers` | Creates the ResNet layers based on the configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes ResNet network-specific data from a binary reader. |
| `ForTesting(Int32,Int32)` | Creates a minimal ResNet network optimized for fast test execution. |
| `Forward(Tensor<>)` | Performs a forward pass through the ResNet network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the ResNet network model. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `InitializeLayers` | Initializes the layers of the ResNet network based on the configuration. |
| `PredictEager(Tensor<>)` | Makes a prediction using the ResNet network for the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes ResNet network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the ResNet network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |
| `ValidateArchitectureMatchesConfiguration(NeuralNetworkArchitecture<>,ResNetConfiguration)` | Validates that the architecture parameters match the configuration. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_configuration` | The ResNet configuration specifying the variant and parameters. |
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

