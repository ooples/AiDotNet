---
title: "RadialBasisFunctionNetwork<T>"
description: "Represents a Radial Basis Function Network, which is a type of neural network that uses radial basis functions as activation functions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Radial Basis Function Network, which is a type of neural network that uses radial basis functions as activation functions.

## For Beginners

A Radial Basis Function Network is a special type of neural network designed to learn patterns differently than standard networks.

Think of it like a weather prediction system:

- Traditional neural networks might look at many factors and gradually learn weather patterns
- An RBFN instead has "weather experts" (the radial basis functions) who each specialize in a specific weather pattern
- When new data comes in, each expert reports how similar the current conditions are to their specialty
- The network combines these similarity reports to make a prediction

For example, one expert might specialize in "sunny summer days," another in "rainy spring mornings," and so on.
When given new weather data, they each report how close it matches their expertise, and the network uses these
reports to predict the weather.

RBFNs are particularly good at:

- Learning complex patterns quickly
- Function approximation (finding the shape of unknown mathematical functions)
- Classification problems (determining which category something belongs to)

## How It Works

A Radial Basis Function Network (RBFN) is a specialized type of neural network that uses radial basis functions as
activation functions. Unlike traditional neural networks, RBFNs typically have only one hidden layer with a non-linear
radial basis function, followed by a linear output layer. This architecture makes them particularly effective for
function approximation, classification, and systems control problems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RadialBasisFunctionNetwork` | Initializes a new instance of the `RadialBasisFunctionNetwork` class with the specified architecture and radial basis function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `_hiddenSize` | Gets or sets the size of the hidden layer (number of radial basis functions). |
| `_inputSize` | Gets or sets the size of the input layer (number of input features). |
| `_outputSize` | Gets or sets the size of the output layer (number of output values). |
| `_radialBasisFunction` | Gets or sets the radial basis function used in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the radial basis function network with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Radial Basis Function Network. |
| `GetModelMetadata` | Gets metadata about the Radial Basis Function Network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture or default configuration. |
| `PredictCore(Tensor<>)` | Makes a prediction using the current state of the Radial Basis Function Network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Radial Basis Function Network. |
| `Train(Tensor<>,Tensor<>)` | Trains the Radial Basis Function Network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the radial basis function network layers. |

