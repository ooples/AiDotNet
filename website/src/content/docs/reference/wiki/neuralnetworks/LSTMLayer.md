---
title: "LSTMLayer<T>"
description: "Represents a Long Short-Term Memory (LSTM) layer for processing sequential data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Long Short-Term Memory (LSTM) layer for processing sequential data.

## For Beginners

An LSTM layer is like a smart memory system for your AI.

Think of it like a notepad with special features:

- It can remember important information for a long time (unlike simpler neural networks)
- It can forget irrelevant details (using its "forget gate")
- It can decide what new information to write down (using its "input gate")
- It can decide what information to share (using its "output gate")

LSTMs are great for:

- Text generation and language understanding
- Time series prediction (like stock prices)
- Speech recognition
- Any task where the order and context of information matters

For example, when processing the sentence "The clouds are in the ___", an LSTM would remember
that "clouds" appeared earlier, helping it predict "sky" as the missing word.

## How It Works

The LSTM layer is a specialized type of recurrent neural network (RNN) that is designed to capture long-term
dependencies in sequential data. It uses a cell state and a series of gates (forget, input, and output) to control
the flow of information through the network, allowing it to remember important patterns over long sequences while
forgetting irrelevant information.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMLayer(Int32,IActivationFunction<>,IActivationFunction<>)` | Initializes a new lazy `LSTMLayer`: input feature size is resolved from `input.Shape[^1]` on first `Tensor{`; weight tensors and biases are allocated then. |
| `LSTMLayer(Int32,IVectorActivationFunction<>,IVectorActivationFunction<>)` | Lazy ctor with vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BiasC` | Gets the cell gate bias for weight loading. |
| `BiasF` | Gets the forget gate bias for weight loading. |
| `BiasI` | Gets the input gate bias for weight loading. |
| `BiasO` | Gets the output gate bias for weight loading. |
| `Gradients` | Gets a dictionary containing the gradients for all trainable parameters after a backward pass. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |
| `WeightsCh` | Gets the cell gate hidden weights for weight loading. |
| `WeightsCi` | Gets the cell gate input weights for weight loading. |
| `WeightsFh` | Gets the forget gate hidden weights for weight loading. |
| `WeightsFi` | Gets the forget gate input weights for weight loading. |
| `WeightsIh` | Gets the input gate hidden weights for weight loading. |
| `WeightsIi` | Gets the input gate input weights for weight loading. |
| `WeightsOh` | Gets the output gate hidden weights for weight loading. |
| `WeightsOi` | Gets the output gate input weights for weight loading. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActivateTensorConditional(IVectorActivationFunction<>,IActivationFunction<>,Tensor<>)` | Applies activation to a tensor using either vector or scalar activation functions based on configuration. |
| `BuildLstmOptimizerState(String)` | Builds the optimizer state for a specific LSTM parameter. |
| `ClearGpuTrainingCache` | Clears the cached GPU tensors used for training. |
| `CopyLastTimestepHidden(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Copies the final timestep's hidden state from a `[B, seq, hidden]` tensor into a `[B, hidden]` destination so consumers reading `LastHiddenState` after the fused fast path see the same value the per-step loop would have stored. |
| `Deserialize(BinaryReader)` | Deserializes the LSTM layer's parameters from a binary stream. |
| `EnsureInitialized` | Allocates the 8 input/recurrent weight tensors and 4 gate biases using the resolved `_inputSize` and `_hiddenSize`. |
| `EnsureLstmOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures GPU optimizer state buffers exist for all LSTM parameters. |
| `Forward(Tensor<>)` | Performs the forward pass of the LSTM layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass using GPU-accelerated LSTM operations. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the LSTM layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeBias(Tensor<>)` | Initializes a bias tensor with zeros. |
| `InitializeWeight(Tensor<>,)` | Initializes a weight tensor with scaled random values. |
| `InitializeWeights` | Initializes the weights of the LSTM layer. |
| `InvalidateCpuStackedWeights` | Drops the cached CPU stacked weights so the next fused forward repacks them. |
| `InvalidateGpuStackedWeights` | Invalidates and disposes the stacked GPU weight buffers. |
| `MatrixToTensor(Matrix<>)` | Converts a Matrix to a 2D Tensor for use in computation graphs. |
| `OnFirstForward(Tensor<>)` | Resolves `_inputSize` from `input.Shape[^1]` and propagates the full input shape into the layer's resolved input/output shapes (output preserves batch / time dims, last axis becomes `_hiddenSize`). |
| `PrepareStackedWeightsForGpu(IDirectGpuBackend)` | Prepares stacked weights in PyTorch format (i, f, g, o order) for the fused LSTM kernel. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Serializes the LSTM layer's parameters to a binary stream. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the LSTM layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SetTrainingMode(Boolean)` | Resets the internal state of the LSTM layer. |
| `TryFusedLstmForward(CpuEngine,Tensor<>,Int32,Int32)` | Inference-only helper that packs this layer's 8 split weight tensors (F/I/C/O × i/h) into the concatenated `[4*hidden, in]` / `[4*hidden, hidden]` layout PyTorch `nn.LSTM` uses, then calls `CpuEngine.LstmSequenceForward<float>` (the fused p… |
| `TryFusedLstmForwardTraining(CpuEngine,Tensor<>,Int32,Int32)` | Training counterpart of `Int32)`. |
| `UnstackGradients(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[],Single[])` | Extracts per-gate gradients from stacked gradient buffers after fused backward kernel. |
| `UpdateParameters()` | Updates the parameters of the LSTM layer based on the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update with polymorphic optimizer support. |
| `VectorToTensor(Vector<>)` | Converts a Vector to a 1D Tensor for use in computation graphs. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biasC` | Bias for the cell state candidate. |
| `_biasF` | Bias for the forget gate. |
| `_biasI` | Bias for the input gate. |
| `_biasO` | Bias for the output gate. |
| `_cachedCellStates` | Cached cell states for all time steps (Batch, Time, Hidden). |
| `_cachedHiddenStates` | Cached hidden states for all time steps (Batch, Time, Hidden). |
| `_hiddenSize` | The size of the hidden state and cell state (number of LSTM units). |
| `_inputSize` | The size of each input vector (number of features). |
| `_isInitialized` | True once the first forward has resolved `_inputSize` and allocated the 8 weight tensors + 4 biases. |
| `_lastCellState` | The cell state from the last forward pass. |
| `_lastHiddenState` | The hidden state from the last forward pass. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_sigmoidActivation` | The sigmoid activation function for element-wise operations. |
| `_sigmoidVectorActivation` | The sigmoid activation function for vector operations. |
| `_tanhActivation` | The tanh activation function for element-wise operations. |
| `_tanhVectorActivation` | The tanh activation function for vector operations. |
| `_useVectorActivation` | Flag indicating whether to use vector or scalar activation functions. |
| `_weightsCh` | Weights for the cell state hidden state connections. |
| `_weightsCi` | Weights for the cell state input connections. |
| `_weightsFh` | Weights for the forget gate hidden state connections. |
| `_weightsFi` | Weights for the forget gate input connections. |
| `_weightsIh` | Weights for the input gate hidden state connections. |
| `_weightsIi` | Weights for the input gate input connections. |
| `_weightsOh` | Weights for the output gate hidden state connections. |
| `_weightsOi` | Weights for the output gate input connections. |

