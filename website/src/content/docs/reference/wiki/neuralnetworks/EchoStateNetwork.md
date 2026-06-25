---
title: "EchoStateNetwork<T>"
description: "Represents an Echo State Network (ESN), a type of recurrent neural network with a sparsely connected hidden layer called a reservoir."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an Echo State Network (ESN), a type of recurrent neural network with a sparsely connected hidden layer called a reservoir.

## For Beginners

An Echo State Network is like a pool of water that creates ripples from your input.

Think of it this way:

- You drop a stone into a pool of water (your input)
- The stone creates ripples that bounce around and interact in complex ways (the reservoir)
- Someone watches the pattern of ripples and learns to predict what comes next (the output layer)
- Only the person watching and predicting is trained - the water itself doesn't change how it ripples

This approach is particularly good for processing sequences, like speech or time series data,
because the ripples in the reservoir naturally capture patterns over time without needing
complex training procedures.

## How It Works

An Echo State Network is a unique type of recurrent neural network where the connections between neurons in
the hidden layer (called the reservoir) are randomly generated and remain fixed during training. Only the
output connections from the reservoir to the output layer are trained. The reservoir acts as a dynamic
memory that transforms inputs into high-dimensional representations, enabling the network to process
temporal patterns effectively. The key characteristic of ESNs is the "echo state property" which ensures
that the effect of initial conditions gradually fades away.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EchoStateNetwork` | Initializes a new instance of the `EchoStateNetwork` class with vector activation functions. |
| `EchoStateNetwork(NeuralNetworkArchitecture<>,Int32,Double,Double,Double,Double,Int32,ILossFunction<>,IActivationFunction<>,IActivationFunction<>,IActivationFunction<>,IActivationFunction<>,EchoStateNetworkOptions)` | Initializes a new instance of the `EchoStateNetwork` class with scalar activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` | Makes a prediction using the Echo State Network. |
| `_outputScalarActivation` | Gets or sets the scalar activation function applied to individual elements in the output layer. |
| `_outputVectorActivation` | Gets or sets the vector activation function applied to the output layer. |
| `_reservoirInputScalarActivation` | Gets or sets the scalar activation function applied to individual elements in the input-to-reservoir connections. |
| `_reservoirInputVectorActivation` | Gets or sets the vector activation function applied to the input-to-reservoir connections. |
| `_reservoirOutputScalarActivation` | Gets or sets the scalar activation function applied to individual elements in the reservoir-to-output connections. |
| `_reservoirOutputVectorActivation` | Gets or sets the vector activation function applied to the reservoir-to-output connections. |
| `_reservoirScalarActivation` | Gets or sets the scalar activation function applied to individual elements within the reservoir. |
| `_reservoirVectorActivation` | Gets or sets the vector activation function applied within the reservoir. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSpectralRadius(Matrix<>)` | Calculates the spectral radius of a matrix using the power method. |
| `ComputeInverse(Matrix<>)` | Computes the inverse of a matrix using Gaussian elimination. |
| `ComputeOutput` | Computes the output based on the current reservoir state. |
| `CreateNewInstance` | Creates a new instance of the EchoStateNetwork with the same configuration as the current instance. |
| `DeserializeMatrix(BinaryReader)` | Deserializes a matrix from a binary reader. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Echo State Network-specific data from a binary reader. |
| `DeserializeVector(BinaryReader)` | Deserializes a vector from a binary reader. |
| `FinalizeTraining` | Finalizes training by computing the optimal output weights. |
| `GetModelMetadata` | Gets metadata about the Echo State Network model. |
| `GetOptions` |  |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the layers of the Echo State Network based on the architecture. |
| `InitializeWeights` | Initializes the weights and reservoir state. |
| `PredictSequence(List<Tensor<>>,Boolean)` | Processes a sequence of inputs through the Echo State Network. |
| `ResetReservoirState` | Resets the reservoir state to zeros. |
| `ScaleToSpectralRadius(Matrix<>,Double)` | Scales a matrix to achieve the desired spectral radius. |
| `SerializeMatrix(BinaryWriter,Matrix<>)` | Serializes a matrix to a binary writer. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Echo State Network-specific data to a binary writer. |
| `SerializeVector(BinaryWriter,Vector<>)` | Serializes a vector to a binary writer. |
| `SetLeakingRate(Double)` | Sets the leaking rate for the reservoir. |
| `SetParameters(Vector<>)` |  |
| `SetRegularization(Double)` | Sets the regularization parameter for ridge regression. |
| `SetWarmupPeriod(Int32)` | Sets the warmup period for discarding initial transient reservoir states. |
| `SettleReservoirState(Vector<>)` | Resets the reservoir and drives the (static) input until the reservoir settles onto its input-driven fixed point, leaving the result in `_currentState`. |
| `SolveLinearSystemDouble(Double[0:,0:],Double[0:,0:])` | Solves the symmetric positive-definite system `A · W = B` in double precision via Gauss-Jordan elimination with partial pivoting, returning `W` (or `null` if A is singular). |
| `SolveReadoutRidgeRegression` | Solves the closed-form ridge regression for `_outputWeights` (and `_outputBias`) given the currently collected reservoir states and targets. |
| `Train(Tensor<>,Tensor<>)` | Trains the Echo State Network on a single batch of data. |
| `UpdateParameters(Vector<>)` | Updates the output layer parameters (weights and biases) of the Echo State Network. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that the custom layers form a valid Echo State Network structure. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_collectedStates` | Collected states during training for regression. |
| `_collectedTargets` | Collected targets during training for regression. |
| `_currentState` | The current state of the reservoir. |
| `_inputSize` | Input dimension size. |
| `_inputWeights` | The weight matrix for input-to-reservoir connections. |
| `_isTraining` | Indicates whether the network is being trained. |
| `_leakingRate` | Leaking rate for controlling the update speed of reservoir neurons. |
| `_outputBias` | The bias vector for the output layer. |
| `_outputSize` | Output dimension size. |
| `_outputWeights` | The weight matrix for reservoir-to-output connections. |
| `_random` | Random number generator for initialization. |
| `_regularization` | Regularization parameter for ridge regression during training. |
| `_reservoirBias` | The bias vector for the reservoir. |
| `_reservoirSize` | Gets the size of the reservoir (number of neurons in the hidden layer). |
| `_reservoirState` | Gets or sets the current state of the reservoir. |
| `_reservoirWeights` | The weight matrix for reservoir-to-reservoir connections. |
| `_sparsity` | Gets the sparsity level of connections in the reservoir. |
| `_spectralRadius` | Gets the spectral radius that controls the dynamics of the reservoir. |
| `_warmupPeriod` | Warmup period for discarding initial transient reservoir states during training. |

