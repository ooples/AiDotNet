---
title: "MixtureOfExpertsNeuralNetwork<T>"
description: "Represents a Mixture-of-Experts (MoE) neural network that routes inputs through multiple specialist networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Mixture-of-Experts (MoE) neural network that routes inputs through multiple specialist networks.

## For Beginners

Mixture-of-Experts is like having a team of specialists rather than one generalist.

Imagine you're running a hospital:

- Instead of one doctor handling everything, you have specialists (cardiologist, neurologist, etc.)
- A triage system (gating network) decides which specialist(s) should see each patient
- Each specialist only handles cases they're best suited for

In a MoE neural network:

- Multiple "expert" networks specialize in different patterns in your data
- A "gating network" learns to route each input to the best expert(s)
- Only a few experts process each input (sparse activation), making it efficient
- The final prediction combines the outputs from the selected experts

This model automatically implements IFullModel, allowing it to work with AiModelBuilder
just like any other neural network in AiDotNet.

## How It Works

A Mixture-of-Experts neural network employs multiple expert networks and a gating mechanism to
route inputs to the most appropriate experts. This architecture enables:

- Increased model capacity without proportional compute cost (sparse activation)
- Specialization of different experts on different aspects of the problem
- Improved scalability for large-scale problems

The architecture consists of:

- Multiple expert networks (can be feed-forward, convolutional, etc.)
- A gating/routing network that learns to select appropriate experts
- Optional load balancing loss to ensure all experts are utilized

**Key Features:**

- Configurable number of expert networks
- Top-K sparse routing for computational efficiency
- Automatic load balancing to prevent expert collapse
- Integration with AiModelBuilder for easy training
- Full support for serialization and deserialization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixtureOfExpertsNeuralNetwork` | Initializes a new instance of the MixtureOfExpertsNeuralNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` | Creates a new instance of the MixtureOfExpertsNeuralNetwork with the same configuration as the current instance. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Mixture-of-Experts network-specific data from a binary reader. |
| `Forward(Tensor<>)` | Performs a forward pass through the network with the given input tensor. |
| `GetModelMetadata` | Retrieves metadata about the Mixture-of-Experts neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture and options. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Mixture-of-Experts network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Mixture-of-Experts network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Mixture-of-Experts network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to calculate the error between predicted and expected outputs. |
| `_moeLayer` | The core Mixture-of-Experts layer. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |
| `_options` | Configuration options for the MoE network. |

