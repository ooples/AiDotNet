---
title: "LiquidStateMachine<T>"
description: "Represents a Liquid State Machine (LSM), a type of reservoir computing neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Liquid State Machine (LSM), a type of reservoir computing neural network.

## For Beginners

A Liquid State Machine is a neural network inspired by how real brains process information over time.

Think of it like dropping different objects into a pool of water:

- Each object creates unique ripple patterns when it hits the water
- The ripples interact with each other in complex ways
- By observing these ripple patterns, you can determine what objects were dropped in

In a Liquid State Machine:

- The "reservoir" is like the pool of water with randomly connected neurons
- Input signals create "ripples" of activity through the connected neurons
- The network learns to recognize patterns in how these ripples develop over time

LSMs are particularly good at:

- Processing sequential data (like speech or sensor readings)
- Handling inputs that change over time
- Working with noisy or incomplete information
- Learning temporal patterns without needing extensive training

## How It Works

A Liquid State Machine is a form of reservoir computing that uses a recurrent neural network with 
randomly connected neurons (the "reservoir") to process temporal information. The reservoir transforms 
input signals into higher-dimensional representations, which are then processed by trained readout 
functions. LSMs are particularly effective for processing time-varying inputs and are inspired by 
the dynamics of biological neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiquidStateMachine` | Initializes a new instance of the `LiquidStateMachine` class with the specified architecture and parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the Liquid State Machine with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Liquid State Machine-specific data from a binary reader. |
| `GetModelMetadata` | Gets metadata about the Liquid State Machine model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Liquid State Machine based on the provided architecture. |
| `PredictCore(Tensor<>)` | Performs a forward pass through the Liquid State Machine to make a prediction. |
| `ResetState` | Resets the state of the Liquid State Machine. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Liquid State Machine-specific data to a binary writer. |
| `SetTrainingMode(Boolean)` | Sets the training mode for the Liquid State Machine. |
| `SimulateTimeSeries(List<Tensor<>>)` | Simulates the LSM with time-series data, allowing the reservoir state to evolve over time. |
| `Train(Tensor<>,Tensor<>)` | Trains the Liquid State Machine on a single input-output pair. |
| `TrainCore(Tensor<>,Tensor<>,Boolean)` | Core training step with explicit reservoir-reset control. |
| `TrainOnTimeSeries(List<Tensor<>>,List<Tensor<>>)` | Performs online learning for time-series data, updating the network after each time step. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network using the provided parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_connectionProbability` | Gets the probability of connection between neurons in the reservoir. |
| `_inputScaling` | Gets the scaling factor applied to input signals. |
| `_leakingRate` | Gets the leaking rate of the reservoir neurons. |
| `_reservoirSize` | Gets the size of the reservoir (number of neurons). |
| `_spectralRadius` | Gets the spectral radius of the reservoir weight matrix. |

