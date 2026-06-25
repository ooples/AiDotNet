---
title: "GRULayer<T>"
description: "Represents a Gated Recurrent Unit (GRU) layer for processing sequential data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Gated Recurrent Unit (GRU) layer for processing sequential data.

## For Beginners

This layer helps neural networks understand sequences of data, like sentences or time series.

Think of the GRU as having a "memory" that helps it understand context:

- When reading a sentence, it remembers important words from earlier
- When analyzing stock prices, it remembers relevant trends from previous days
- It uses special "gates" to decide what information to keep or forget

For example, in the sentence "The clouds were dark and it started to ___", 
the GRU would recognize the context and predict "rain" because it remembers
the earlier words about dark clouds.

GRUs are simpler versions of LSTMs (Long Short-Term Memory) but often perform similarly well
while being more efficient to train.

## How It Works

The GRU (Gated Recurrent Unit) layer is a type of recurrent neural network layer that is designed to
capture dependencies over time in sequential data. It addresses the vanishing gradient problem that
standard recurrent neural networks face when dealing with long sequences. The GRU uses update and reset
gates to control the flow of information, allowing the network to retain relevant information over
many time steps while forgetting irrelevant details.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GRULayer(Int32,Boolean,IActivationFunction<>,IActivationFunction<>,Boolean)` | Initializes a lazy `GRULayer`: input feature size is resolved from `input.Shape[^1]` on first `Tensor{`; weight tensors and biases are allocated then. |
| `GRULayer(Int32,IVectorActivationFunction<>,Boolean,IVectorActivationFunction<>,Boolean)` | Lazy ctor with vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActivateTape(Tensor<>,IVectorActivationFunction<>,IActivationFunction<>,Boolean)` | Applies the appropriate activation function to the input tensor. |
| `ApplyActivationDerivative(Tensor<>,Boolean)` | Applies the derivative of the appropriate activation function to the input tensor. |
| `BroadcastVector(Tensor<>,Int32)` | Broadcasts a 1D tensor across the batch dimension. |
| `BuildGruOptimizerState(String)` | Builds optimizer state for a specific parameter tensor. |
| `ClearGpuTrainingCache` | Clears the GPU training cache to release GPU memory. |
| `Clone` | Creates a deep copy of this GRU layer with independent weights and reset state. |
| `ComputeGate(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Boolean)` | Computes a gate activation for the GRU layer. |
| `CreateOnesLike(Tensor<>)` | Creates a tensor of ones with the same shape as the input tensor. |
| `EnsureGruOptimizerState(IGpuOptimizerConfig,IDirectGpuBackend)` | Ensures optimizer state tensors are allocated for the given optimizer type. |
| `EnsureInitialized` | Allocates the 6 GRU weight tensors and 3 gate biases using resolved `_inputSize` and `_hiddenSize`. |
| `Forward(Tensor<>)` | Performs the forward pass of the GRU layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors using fused sequence kernel. |
| `GetMetadata` | Gets all trainable parameters of the layer as a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor(Int32,Int32,)` | Initializes a tensor with scaled random values. |
| `InvalidateGpuStackedWeights` | Invalidates and disposes the stacked GPU weight buffers. |
| `OnFirstForward(Tensor<>)` | Resolves `_inputSize` from `input.Shape[^1]` on first forward. |
| `PrepareStackedWeightsForGpu(IDirectGpuBackend)` | Prepares stacked weights in PyTorch format (r, z, n order) for the fused GRU kernel. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UnstackGradients(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[])` | Extracts per-gate gradients from stacked gradient buffers after fused backward kernel. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the layer with the given vector of parameter values. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update with polymorphic optimizer support. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_Uh` | The weight tensors that transform the previous hidden state. |
| `_Ur` | The weight tensors that transform the previous hidden state. |
| `_Uz` | The weight tensors that transform the previous hidden state. |
| `_Wh` | The weight tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_Wr` | The weight tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_Wz` | The weight tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_activation` | The computation engine (CPU or GPU) for vectorized operations. |
| `_allHiddenStates` | All hidden states from the last forward pass, used when returning sequences. |
| `_bh` | The bias tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_br` | The bias tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_bz` | The bias tensors for the update gate (z), reset gate (r), and candidate hidden state (h). |
| `_dUh` | Gradients for the weight matrices during backpropagation. |
| `_dUr` | Gradients for the weight matrices during backpropagation. |
| `_dUz` | Gradients for the weight matrices during backpropagation. |
| `_dWh` | Gradients for the weight matrices during backpropagation. |
| `_dWr` | Gradients for the weight matrices during backpropagation. |
| `_dWz` | Gradients for the weight matrices during backpropagation. |
| `_dbh` | Gradients for the bias vectors during backpropagation. |
| `_dbr` | Gradients for the bias vectors during backpropagation. |
| `_dbz` | Gradients for the bias vectors during backpropagation. |
| `_hiddenSize` | The size of the hidden state vector. |
| `_inputSize` | The size of the input feature vector at each time step. |
| `_isInitialized` | True once the first forward has resolved `_inputSize` and allocated the 6 weight tensors + 3 biases. |
| `_lastH` | The activation values for the update gate (z), reset gate (r), and candidate hidden state (h) from the last forward pass. |
| `_lastHiddenState` | The final hidden state from the last forward pass. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastR` | The activation values for the update gate (z), reset gate (r), and candidate hidden state (h) from the last forward pass. |
| `_lastZ` | The activation values for the update gate (z), reset gate (r), and candidate hidden state (h) from the last forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_recurrentActivation` | The activation function applied to the update and reset gates. |
| `_returnSequences` | Determines whether the layer returns the full sequence of hidden states or just the final state. |
| `_stateful` | When true, the hidden state carries over between successive Forward calls (Keras-style `stateful=True`) instead of resetting to zeros each call. |
| `_vectorActivation` | The vector activation function applied to the candidate hidden state. |
| `_vectorRecurrentActivation` | The vector activation function applied to the update and reset gates. |

