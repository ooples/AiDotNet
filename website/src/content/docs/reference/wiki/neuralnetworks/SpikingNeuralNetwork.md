---
title: "SpikingNeuralNetwork<T>"
description: "Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics.

## For Beginners

Spiking Neural Networks (SNNs) work like real biological neurons:
they communicate using timed electrical pulses (spikes) rather than continuous values. Each
neuron accumulates input over time and only fires when a threshold is reached, then resets.
This temporal coding makes SNNs extremely energy-efficient on neuromorphic hardware and
naturally suited for processing time-varying signals like sensor data and event cameras.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpikingNeuralNetwork` | Initializes a new instance of the `SpikingNeuralNetwork` class with the specified architecture and a vector activation function. |
| `SpikingNeuralNetwork(NeuralNetworkArchitecture<>,Double,Int32,IActivationFunction<>,ILossFunction<>,SpikingNeuralNetworkOptions)` | Initializes a new instance of the `SpikingNeuralNetwork` class with the specified architecture and a scalar activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |
| `_scalarActivation` | Gets or sets the scalar activation function used in the network. |
| `_simulationSteps` | Gets or sets the number of time steps to simulate when processing input. |
| `_timeStep` | Gets or sets the simulation time step for the spiking neural network. |
| `_vectorActivation` | Gets or sets the vector activation function used in the network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateSpikeTrainToOutput(List<Vector<>>)` | Aggregates a spike train (sequence of spikes over time) into a final output vector. |
| `ApplySTDPLearning(List<List<Vector<>>>,List<List<Vector<>>>,Vector<>,Int32)` | Applies Spike-Timing-Dependent Plasticity (STDP) learning based on spike history. |
| `CalculateSTDPWeightChange(List<Vector<>>,List<Vector<>>,Int32,Int32,Int32)` | Calculates weight change based on STDP rule for a specific connection. |
| `CreateNewInstance` | Creates a new instance of the Spiking Neural Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes SNN-specific data from a binary reader. |
| `FindReadoutBoundary` | Returns the index of the first non-spiking layer in the stack (i.e. |
| `GetModelMetadata` | Gets metadata about the spiking neural network. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture or default configuration. |
| `InitializeNeuronStates` | Initializes the internal neuron states for simulation. |
| `PredictCore(Tensor<>)` | Makes a prediction using the spiking neural network. |
| `ReadFirstShapeAxis(Int32[])` | Safe-indexed first-axis read for lazy layers that may return an empty shape array before resolution. |
| `ResetState` | Resets the internal state of the network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes SNN-specific data to a binary writer. |
| `SetLayerThresholds(Int32,Vector<>)` | Sets custom firing thresholds for neurons in a specific layer. |
| `SetNeuronModelParameters(,Int32)` | Sets the neuron model parameters for the network. |
| `SetSimulationParameters(Double,Int32)` | Sets the simulation parameters for the network. |
| `Train(Tensor<>,Tensor<>)` | Trains the spiking neural network on input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the spiking neural network layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_firingThresholds` | Firing thresholds for neurons in each layer. |
| `_membraneDecay` | Gets or sets the decay constant for neuron membrane potentials. |
| `_membranePotentials` | Stores the membrane potentials for all neurons in the network. |
| `_refractoryCounters` | Tracks the refractory state of each neuron. |
| `_refractoryPeriod` | Gets or sets the refractory period for neurons after firing. |

