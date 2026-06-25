---
title: "RestrictedBoltzmannMachine<T>"
description: "Represents a Restricted Boltzmann Machine, which is a type of neural network that learns probability distributions over its inputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Restricted Boltzmann Machine, which is a type of neural network that learns probability distributions over its inputs.

## For Beginners

A Restricted Boltzmann Machine is like a two-way translator between data and features.

Think of it like this:

- The visible layer is like words in English
- The hidden layer is like words in French
- The network learns how to translate back and forth between the languages

When you train an RBM:

- It learns to recognize patterns in your data (translate English to French)
- It also learns to recreate the original data from those patterns (translate French back to English)

For example, if you train an RBM on images of faces:

- The visible layer represents the pixel values of the images
- The hidden layer might learn to recognize features like "has a mustache" or "is smiling"
- Once trained, you could activate certain hidden units to generate new face images with specific features

RBMs can be used for dimensionality reduction, feature learning, pattern completion, and even generating
new data samples similar to the training data.

## How It Works

A Restricted Boltzmann Machine (RBM) is a two-layer neural network that learns to reconstruct its input data.
Unlike feedforward networks, RBMs are generative models that learn the probability distribution of the training data.
They consist of a visible layer (representing the input data) and a hidden layer (representing features), with
connections between layers but no connections within a layer (hence "restricted"). RBMs are trained using an
algorithm called Contrastive Divergence, which involves both forward and backward passes between layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RestrictedBoltzmannMachine(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,IVectorActivationFunction<>,ILossFunction<>,RestrictedBoltzmannMachineOptions)` | Initializes a new instance of the `RestrictedBoltzmannMachine` class with the specified architecture, sizes, and vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenSize` | Gets the number of neurons in the hidden layer. |
| `ParameterCount` | Gets the total number of parameters (weights and biases) in the RBM. |
| `VisibleSize` | Gets the number of neurons in the visible layer. |
| `_hiddenBiases` | Gets or sets the bias values for the hidden layer neurons. |
| `_scalarActivation` | Gets or sets the scalar activation function used in the RBM. |
| `_vectorActivation` | Gets or sets the vector activation function used in the RBM. |
| `_visibleBiases` | Gets or sets the bias values for the visible layer neurons. |
| `_weights` | Gets or sets the weight matrix representing connections between visible and hidden neurons. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAssociations(Tensor<>,Tensor<>)` | Computes the outer product of visible and hidden activations to get association matrix. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes RBM-specific data from a binary reader. |
| `ExtractFeatures(Tensor<>,Boolean)` | Extracts features from input data using the trained RBM. |
| `GenerateSamples(Int32,Int32)` | Generates samples from the RBM by starting with a random visible state and performing Gibbs sampling. |
| `GetHiddenLayerActivation(Tensor<>)` | Calculates the activation probabilities of the hidden layer given the visible layer. |
| `GetModelMetadata` | Gets metadata about the RBM model. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetParameterChunks` | Yields the RBM's trainable parameters (weights matrix and the two bias vectors) as `Tensor` chunks so test infrastructure that walks `GetParameterChunks` can observe post-train parameter changes. |
| `GetParameters` | Updates the parameters of the RBM. |
| `GetVisibleLayerActivation(Tensor<>)` | Calculates the activation probabilities of the visible layer given the hidden layer. |
| `InitializeHintonWeights(Int32,Int32)` | Initializes weights per Hinton 2006 ("A Practical Guide to Training Restricted Boltzmann Machines"): `w ~ N(0, 0.01²)`. |
| `InitializeLayers` | Initializes the neural network layers. |
| `InitializeParameters` | Initializes the weights and biases of the RBM with appropriate starting values. |
| `PredictCore(Tensor<>)` | Makes predictions using the RBM by computing hidden layer activations. |
| `SampleBinaryStates(Tensor<>)` | Samples binary states from activation probabilities. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes the RBM-specific data to a binary writer. |
| `SetTrainingParameters(,Int32)` | Sets the training parameters for the RBM. |
| `Train(Tensor<>,Tensor<>)` | Trains the RBM using Contrastive Divergence. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultVisibleSize` | Initializes a new instance of the `RestrictedBoltzmannMachine` class with the specified architecture, sizes, and scalar activation function. |
| `_cdSteps` | Gets or sets the number of steps to run the Gibbs sampling chain during Contrastive Divergence. |
| `_learningRate` | Gets or sets the learning rate for Contrastive Divergence training. |

