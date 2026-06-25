---
title: "OccupancyNeuralNetwork<T>"
description: "Represents a Neural Network specialized for occupancy detection and prediction in spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Neural Network specialized for occupancy detection and prediction in spaces.

## For Beginners

Think of this network as a smart system that can "understand" when people
are present in a room or building by analyzing data from various sensors. Just like you might
determine if someone is in a room by noticing changes in temperature, sounds, or movement,
this neural network learns patterns in sensor data that indicate human presence. It's particularly
useful for smart buildings, energy management, security systems, and space utilization analysis.

## How It Works

An Occupancy Neural Network processes sensor data to detect and predict the presence and number
of people in a given space. It can handle various types of sensor inputs including temperature,
humidity, CO2 levels, motion detection, and other environmental factors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OccupancyNeuralNetwork` | Initializes a new instance of the OccupancyNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HistoryWindowSize` | Gets the size of the time window used for temporal data. |
| `IncludesTemporalData` | Gets a value indicating whether this network processes temporal data. |
| `SupportsTraining` |  |
| `_historyWindowSize` | Gets or sets the size of the time window used for temporal data processing. |
| `_includeTemporalData` | Gets or sets a value indicating whether this network processes temporal data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateError(Tensor<>,Tensor<>)` | Calculates the error between predicted and expected outputs. |
| `CreateNewInstance` | Creates a new instance of the OccupancyNeuralNetwork with the same architecture and temporal configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `ForwardTemporal(Tensor<>)` | Performs a forward pass through the network for temporal data. |
| `GetModelMetadata` | Gets metadata about the occupancy neural network. |
| `GetOptions` |  |
| `GetOrCreateBaseOptimizer` | Uses an Adam (AMSGrad) optimizer whose learning rate comes from `BaseOptimizerInitialLearningRate` (default 0.01, vs the framework default of 0.001). |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the occupancy neural network. |
| `ProcessSingleInput(Vector<>)` | Processes a single input vector through the network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the neural network on sensor data and occupancy labels. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the neural network based on computed gradients. |
| `UpdatePrediction(Vector<>,Queue<Vector<>>)` | Processes a new sensor reading and updates the prediction for real-time occupancy detection. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_internalSensorHistory` | Buffer to store historical sensor readings for temporal processing. |

