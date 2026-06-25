---
title: "DeepBoltzmannMachine<T>"
description: "Represents a Deep Boltzmann Machine (DBM), a hierarchical generative model consisting of multiple layers of stochastic neurons."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Deep Boltzmann Machine (DBM), a hierarchical generative model consisting of multiple layers of stochastic neurons.

## For Beginners

A Deep Boltzmann Machine is like a multi-story pattern detector.

Think of it as a series of layers, each learning increasingly abstract patterns:

- The visible layer represents the raw data (e.g., pixel values in an image)
- The first hidden layer might learn simple patterns (e.g., edges, corners)
- Higher hidden layers learn more complex patterns (e.g., shapes, objects)
- The deeper the network, the more abstract the patterns it can learn

For example, in an image recognition system:

- Layer 1 might detect edges and basic textures
- Layer 2 might combine these into simple shapes
- Layer 3 might recognize more complex objects

DBMs can both recognize patterns in data and generate new data with similar patterns.

## How It Works

A Deep Boltzmann Machine is an extension of the Restricted Boltzmann Machine to multiple hidden layers.
It consists of a visible layer and multiple hidden layers with connections between adjacent layers but no connections
within the same layer. DBMs are used for unsupervised learning, feature extraction, and generative modeling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepBoltzmannMachine` | Initializes a new instance with default settings. |
| `DeepBoltzmannMachine(NeuralNetworkArchitecture<>,Int32,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,IActivationFunction<>,Int32,Int32,DeepBoltzmannMachineOptions)` | Initializes a new instance of the DeepBoltzmannMachine class with scalar activation. |
| `DeepBoltzmannMachine(NeuralNetworkArchitecture<>,Int32,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,IVectorActivationFunction<>,Int32,Int32,DeepBoltzmannMachineOptions)` | Initializes a new instance of the DeepBoltzmannMachine class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Updates the parameters of the DBM with the given vector of parameter values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActivationFunction(Tensor<>)` | Applies the activation function to a tensor. |
| `AdaptInputWeights(Int32)` | Adapts the first layer weights and biases when actual input size differs from architecture. |
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the reconstruction error between the original input and its reconstruction. |
| `CreateNewInstance` | Creates a new instance of the deep boltzmann machine model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Deep Boltzmann Machine-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the Deep Boltzmann Machine model. |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the layers of the neural network. |
| `InitializeParameters` | Initializes the weights and biases of the DBM. |
| `MatMul2D(Tensor<>,Tensor<>)` | Manual 2D matrix multiplication to avoid Engine.TensorMatMul issues with certain shapes. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Deep Boltzmann Machine. |
| `PretrainLayerwise(Tensor<>,Int32,)` | Performs layer-wise pretraining of the DBM using a greedy approach. |
| `PropagateDown(Tensor<>)` | Propagates the deepest hidden layer activation downward through the network to the visible layer. |
| `PropagateToLayer(Tensor<>,Int32)` | Propagates the input to a specific layer in the network. |
| `PropagateUp(Tensor<>)` | Propagates the input upward through the network from visible to hidden layers. |
| `Reconstruct(Tensor<>)` | Reconstructs the input by propagating it up through the hidden layers and back down. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Deep Boltzmann Machine-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Deep Boltzmann Machine on the provided data. |
| `TrainOnBatch(Tensor<>,)` | Trains the DBM on a single batch of data using contrastive divergence. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activationFunction` | Gets or sets the scalar activation function for the network. |
| `_batchSize` | Gets or sets the number of examples processed in each training batch. |
| `_cdSteps` | Gets or sets the number of contrastive divergence steps in training. |
| `_epochs` | Gets or sets the number of training epochs. |
| `_layerBiases` | Gets or sets the bias vectors for each layer in the network. |
| `_layerSizes` | Gets or sets the number of units in each layer of the network. |
| `_layerWeights` | Gets or sets the weight matrices connecting adjacent layers in the network. |
| `_learningRate` | Gets or sets the learning rate for parameter updates. |
| `_learningRateDecay` | Gets or sets the learning rate decay factor per epoch. |
| `_vectorActivationFunction` | Gets or sets the vector activation function for the network. |

