---
title: "HTMNetwork<T>"
description: "Represents a Hierarchical Temporal Memory (HTM) network, a biologically-inspired sequence learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Hierarchical Temporal Memory (HTM) network, a biologically-inspired sequence learning algorithm.

## For Beginners

HTM is a special type of neural network inspired by how the human brain works.

Think of HTM like a system that learns patterns over time:

- It's similar to how your brain recognizes songs or predicts what comes next in a familiar sequence
- It's especially good at learning from time-series data (information that changes over time)
- It can recognize patterns even when they contain noise or slight variations

HTM networks have two main parts:

- Spatial Pooler: This converts incoming data into a special format that highlights important patterns

(like how your brain might focus on the melody of a song rather than background noise)

- Temporal Memory: This learns sequences and can predict what might come next

(like how you can anticipate the next note in a familiar song)

HTM is particularly useful for:

- Anomaly detection (finding unusual patterns)
- Sequence prediction (guessing what comes next)
- Pattern recognition in noisy data

## How It Works

Hierarchical Temporal Memory (HTM) is a machine learning model that mimics the structural and algorithmic properties 
of the neocortex. It is particularly designed for sequence learning, prediction, and anomaly detection in 
time-series data. HTM networks consist of two main components: a Spatial Pooler that creates sparse distributed 
representations of inputs, and a Temporal Memory that learns sequences of these representations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HTMNetwork` | Initializes a new instance of the `HTMNetwork` class with the specified architecture and parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `_cellsPerColumn` | Gets the number of cells per column in the temporal memory. |
| `_columnCount` | Gets the number of columns in the spatial pooler. |
| `_inputSize` | Gets the size of the input vector. |
| `_sparsityThreshold` | Gets the target sparsity for the spatial pooler output. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAnomalyScore` | Calculates the current anomaly score based on the network's state. |
| `CreateNewInstance` | Creates a new instance of the HTM Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes HTM-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the HTM network. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the HTM network based on the provided architecture. |
| `Learn(Vector<>)` | Processes an input vector through the network and updates the network's internal state based on the input. |
| `Predict(Vector<>)` | Makes a prediction using the current state of the HTM network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the HTM network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes HTM-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the HTM network on a sequence of inputs. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network using the provided parameter vector. |

