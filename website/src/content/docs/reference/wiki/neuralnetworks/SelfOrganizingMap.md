---
title: "SelfOrganizingMap<T>"
description: "Represents a Self-Organizing Map, which is an unsupervised neural network that produces a low-dimensional representation of input data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Self-Organizing Map, which is an unsupervised neural network that produces a low-dimensional representation of input data.

## For Beginners

A Self-Organizing Map is like a smart way to arrange data on a map based on similarities.

Think of it like organizing books on a bookshelf:

- You have many books (input data) with different characteristics
- You want to arrange them so similar books are placed near each other
- Over time, you develop a system where sci-fi books are in one section, romance in another, etc.

A SOM works in a similar way:

- It takes complex data with many attributes
- It creates a 2D "map" where each location represents certain characteristics
- Similar data points end up mapped to nearby locations
- Different regions of the map represent different types of data

This is useful for:

- Visualizing complex data with many dimensions
- Finding natural groupings (clusters) in data
- Reducing complex data to simpler patterns
- Discovering relationships that might not be obvious

## How It Works

A Self-Organizing Map (SOM), also known as a Kohonen map, is a type of artificial neural network that
uses unsupervised learning to produce a low-dimensional (typically two-dimensional) representation
of higher-dimensional input data. SOMs preserve the topological properties of the input space, meaning
that similar inputs will be mapped to nearby neurons in the output map. This makes SOMs useful for
visualization, clustering, and dimensionality reduction of complex data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfOrganizingMap` | Initializes a new instance of the `SelfOrganizingMap` class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |
| `_inputDimension` | Gets or sets the dimensionality of the input data. |
| `_mapHeight` | Gets or sets the height of the map (number of neurons vertically). |
| `_mapWidth` | Gets or sets the width of the map (number of neurons horizontally). |
| `_weights` | Gets or sets the weight matrix representing the connection strengths between input dimensions and map neurons. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDistance(Vector<>,Vector<>)` | Calculates the Euclidean distance between two vectors. |
| `CalculateInfluence(,)` | Calculates the influence of the best matching unit on a neuron based on distance. |
| `CalculateLearningRate(,Int32,Int32)` | Calculates the current learning rate based on the initial rate and the current epoch. |
| `CalculateRadius(,Int32,)` | Calculates the current neighborhood radius based on the initial radius and the current epoch. |
| `CalculateWeightDelta(Vector<>,Vector<>,,)` | Calculates the weight delta for updating a neuron's weights. |
| `CreateNewInstance` | Creates a new instance of the Self-Organizing Map with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes the specific data of the Self-Organizing Map. |
| `FindBestMatchingUnit(Vector<>)` | Finds the index of the neuron that best matches the input vector. |
| `GetModelMetadata` | Gets the metadata of the Self-Organizing Map model. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetParameterChunks` | Yields the SOM codebook as a single parameter chunk. |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the neural network layers. |
| `InitializeWeights` | Initializes the weights of the SOM with random values. |
| `PredictCore(Tensor<>)` | Predicts the output for a given input using the trained Self-Organizing Map. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes the specific data of the Self-Organizing Map. |
| `Train(Tensor<>,Tensor<>)` | Trains the Self-Organizing Map using the provided input. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the SOM from a flat parameter vector. |
| `UpdateWeights(Vector<>,Int32,,)` | Updates the weights of neurons in the neighborhood of the best matching unit. |

