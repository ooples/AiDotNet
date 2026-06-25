---
title: "VGGNetwork<T>"
description: "Represents a VGG (Visual Geometry Group) neural network architecture for image classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a VGG (Visual Geometry Group) neural network architecture for image classification.

## For Beginners

VGG networks are one of the foundational architectures in deep learning for
image recognition. Despite being developed in 2014, they remain popular because:

- They're simple to understand - just stacked convolutions and pooling
- They serve as excellent baselines for comparing new architectures
- They're great for transfer learning (using a pre-trained network as a starting point)
- The features they learn are highly transferable to other visual tasks

## How It Works

VGG networks are deep convolutional neural networks developed by the Visual Geometry Group at
Oxford University. They are characterized by their use of small (3x3) convolution filters stacked
in increasing depth, which allows them to learn complex hierarchical features.

**Architecture:** VGG networks consist of:

- Multiple blocks of 3x3 convolutional layers with ReLU activation
- Max pooling (2x2, stride 2) after each block to reduce spatial dimensions
- Optional batch normalization after each convolution (in _BN variants)
- Three fully connected layers (4096 -> 4096 -> num_classes)
- Dropout regularization in the fully connected layers

**Typical Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VGGNetwork` | Initializes a new instance of the VGGNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClasses` | Gets the number of output classes for classification. |
| `UsesBatchNormalization` | Gets whether this network uses batch normalization. |
| `Variant` | Gets the VGG variant being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyParameterUpdates` | Applies parameter updates using the optimizer. |
| `CalculateOutputGradient(Tensor<>,Tensor<>)` | Calculates the gradient of the loss with respect to the network's output. |
| `CreateNewInstance` | Creates a new instance of the VGG network model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes VGG network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the VGG network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the VGG network model. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `InitializeLayers` | Initializes the layers of the VGG network based on the configuration. |
| `PredictEager(Tensor<>)` | Makes a prediction using the VGG network for the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes VGG network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the VGG network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |
| `ValidateArchitectureMatchesConfiguration(NeuralNetworkArchitecture<>,VGGConfiguration)` | Validates that the architecture parameters match the configuration. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_configuration` | The VGG configuration specifying the variant and parameters. |
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |

