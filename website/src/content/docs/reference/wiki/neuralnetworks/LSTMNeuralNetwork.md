---
title: "LSTMNeuralNetwork<T>"
description: "Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing sequential data like text, time series, or audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing
sequential data like text, time series, or audio.

## For Beginners

An LSTM Neural Network is a special type of neural network designed for understanding sequences and patterns that unfold over time.

Think of an LSTM like a smart notepad that can:

- Remember important information for long periods
- Forget irrelevant details
- Update its notes with new information
- Decide what parts of its memory to use for making predictions

For example, when processing a sentence like "The clouds are in the sky", an LSTM can:

- Remember "The clouds" as the subject even after seeing several more words
- Understand that "are" should agree with the plural "clouds" 
- Predict that "sky" might come after "in the" because clouds are typically in the sky

LSTMs are particularly good at:

- Text generation and language modeling
- Speech recognition
- Time series prediction (like stock prices or weather)
- Translation between languages
- Any task where the order of data matters and patterns may span across long sequences

## How It Works

Long Short-Term Memory networks are a special kind of recurrent neural network designed to overcome
the vanishing gradient problem that traditional RNNs face. LSTMs have a complex internal structure 
with specialized "gates" that regulate the flow of information, allowing them to remember patterns
over long sequences and selectively forget irrelevant information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMNeuralNetwork` | Creates a new LSTM Neural Network with customizable loss and activation functions, using scalar activation functions. |
| `LSTMNeuralNetwork(NeuralNetworkArchitecture<>,ILossFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,LSTMOptions)` | Creates a new LSTM Neural Network with customizable loss and activation functions, using vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CellGateActivation` | The activation function to apply to the cell gate. |
| `CellGateVectorActivation` | The activation function to apply to the cell gate. |
| `ForgetGateActivation` | The activation function to apply to the forget gate. |
| `ForgetGateVectorActivation` | The activation function to apply to the forget gate. |
| `InputGateActivation` | The activation function to apply to the input gate. |
| `InputGateVectorActivation` | The activation function to apply to the input gate. |
| `OutputGateActivation` | The activation function to apply to the output gate. |
| `OutputGateVectorActivation` | The activation function to apply to the output gate. |
| `ScalarActivation` | The activation function to apply to cell state outputs. |
| `VectorActivation` | The vector activation function to apply to cell state outputs. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchAndTimeDimensions(Tensor<>)` | Adds both batch and time dimensions to a tensor. |
| `AddBatchDimension(Tensor<>)` | Adds a batch dimension to a tensor. |
| `AddTimeDimension(Tensor<>)` | Adds a time dimension to a tensor. |
| `BackwardLSTMCell(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32)` | Performs the backward pass through a single LSTM cell. |
| `CountLSTMLayers` | Counts the number of LSTM layers in the network. |
| `CreateNewInstance` | Creates a new instance of the LSTM Neural Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes LSTM-specific data from a binary reader. |
| `ExtractMatrix(Vector<>,Int32,Int32,Int32)` | Extracts a matrix from a parameter vector. |
| `ExtractTimeStep(Tensor<>,Int32)` | Extracts a specific time step from a batched sequence. |
| `ExtractVector(Vector<>,Int32,Int32)` | Extracts a vector from a parameter vector. |
| `GetLSTMHiddenSize` | Gets the hidden size for LSTM layers from the architecture or layer configuration. |
| `GetLSTMLayerHiddenSize(Int32)` | Gets the hidden size for a specific LSTM layer. |
| `GetLSTMLayerParameters(Int32)` | Gets the LSTM layer parameters for a specific layer. |
| `GetModelMetadata` | Updates the network parameters based on calculated gradients. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the layers of the LSTM network based on the provided architecture. |
| `InitializeStates(Int32)` | Initializes LSTM states (hidden state and cell state) for all layers. |
| `IsLSTMLayer(ILayer<>)` | Determines if a layer is an LSTM layer or any other type of recurrent layer. |
| `IsZeroTensor(Tensor<>)` | Checks if a tensor contains only zero values. |
| `PredictCore(Tensor<>)` | Processes input through the LSTM network to generate predictions. |
| `ProcessBatchedSequence(Tensor<>)` | Processes a batched sequence through the LSTM network. |
| `ProcessLSTMCell(Tensor<>,Tensor<>,Tensor<>,Int32)` | Processes input through an LSTM cell, applying proper gating mechanisms. |
| `ProcessSequence(Tensor<>)` | Processes a single sequence through the LSTM network. |
| `ProcessSingleSample(Tensor<>)` | Processes a single input sample through the LSTM network. |
| `ProcessSingleTimeStepBatch(Tensor<>)` | Processes a batch of single time step inputs. |
| `ProcessTimeStep(Tensor<>,Dictionary<Int32,ValueTuple<Tensor<>,Tensor<>>>)` | Processes a single time step through all layers of the network. |
| `RemoveBatchAndTimeDimensions(Tensor<>)` | Removes both batch and time dimensions from a tensor. |
| `RemoveBatchDimension(Tensor<>)` | Removes a batch dimension from a tensor. |
| `RemoveTimeDimension(Tensor<>)` | Removes a time dimension from a tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes LSTM-specific data to a binary writer. |
| `StackAlongTimeDimension(List<Tensor<>>)` | Stacks a list of tensors along the time dimension. |
| `StoreLSTMGradients(Int32,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Stores LSTM gradients for parameter updates. |
| `TransformHiddenGradientToOutputGradient(Tensor<>,Int32[])` | Trains the LSTM network on input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the internal parameters (weights and biases) of the network with new values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_storedActivations` | Dictionary to store activations and states from the forward pass. |

