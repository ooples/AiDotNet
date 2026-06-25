---
title: "QuantumNeuralNetwork<T>"
description: "Represents a Quantum Neural Network, which combines quantum computing principles with neural network architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Quantum Neural Network, which combines quantum computing principles with neural network architecture.

## For Beginners

A Quantum Neural Network combines ideas from quantum computing with neural networks.

Think of it like upgrading from a regular calculator to a special calculator with new abilities:

- Regular neural networks use normal bits (0 or 1)
- Quantum neural networks use quantum bits or "qubits" that can be 0, 1, or both at the same time
- This "both at the same time" property (called superposition) gives quantum networks special abilities
- These networks might solve certain problems much faster than regular neural networks

For example, a quantum neural network might find patterns in complex data or optimize solutions
in ways that would be extremely difficult for traditional neural networks.

While the math behind quantum computing is complex, you can think of a quantum neural network
as having the potential to explore many possible solutions simultaneously rather than one at a time.

## How It Works

A Quantum Neural Network (QNN) is a neural network architecture that leverages quantum computing principles
to potentially solve certain problems more efficiently than classical neural networks. It uses quantum bits (qubits)
instead of classical bits, allowing it to process information in ways not possible with traditional neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuantumNeuralNetwork` | Initializes a new instance of the `QuantumNeuralNetwork` class with the specified architecture and number of qubits. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the loss between the predicted output and the expected output. |
| `ConvertToComplexTensor(Tensor<>)` | Converts a real-valued tensor to a complex-valued tensor. |
| `ConvertToComplexTensor(Vector<>)` | Converts a real-valued vector to a complex-valued tensor. |
| `CreateNewInstance` | Creates a new instance of the quantum neural network with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes quantum neural network-specific data from a binary reader. |
| `ExtractRealPart(Tensor<Complex<>>)` | Extracts the real part from a complex-valued tensor. |
| `GetModelMetadata` | Retrieves metadata about the quantum neural network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture or default configuration. |
| `MeasureQuantumState(Tensor<Complex<>>)` | Measures the quantum state to produce a classical output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the quantum neural network for the given input. |
| `PrepareQuantumState(Tensor<>)` | Prepares a quantum state from a classical input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes quantum neural network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the quantum neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the quantum neural network layers. |
| `UpdateQuantumParameters(List<Tensor<>>)` | Updates the parameters of the quantum neural network layers based on calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numQubits` | Gets or sets the number of qubits used in the quantum neural network. |
| `_preprocessingPipeline` | The preprocessing pipeline that handles data transformation for quantum state preparation. |

