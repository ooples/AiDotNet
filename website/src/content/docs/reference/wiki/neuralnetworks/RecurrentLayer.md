---
title: "RecurrentLayer<T>"
description: "Represents a recurrent neural network layer that processes sequential data by maintaining a hidden state."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a recurrent neural network layer that processes sequential data by maintaining a hidden state.

## For Beginners

This layer is designed to work with data that comes in sequences.

Think of the RecurrentLayer as having a memory that helps it understand sequences:

- When reading a sentence word by word, it remembers previous words to understand context
- When analyzing time series data, it remembers past values to predict future trends
- When processing video frames, it remembers earlier frames to track movement

Unlike regular layers that process each input independently, this layer:

- Takes both the current input and its own memory (hidden state) to make decisions
- Updates its memory after seeing each item in the sequence
- Passes this updated memory forward to the next time step

For example, when processing the sentence "The cat sat on the mat":

- At the word "cat", it remembers "The" came before
- At the word "sat", it remembers both "The" and "cat" came before
- This context helps it understand the full meaning of the sentence

This ability to maintain information across a sequence makes recurrent layers 
powerful for tasks involving text, time series, audio, and other sequential data.

## How It Works

The RecurrentLayer implements a basic recurrent neural network (RNN) that processes sequence data by 
maintaining and updating a hidden state over time steps. For each element in the sequence, the layer 
computes a new hidden state based on the current input and the previous hidden state. This allows the 
network to capture temporal dependencies and patterns in sequential data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecurrentLayer(Int32,IActivationFunction<>)` | Lazy ctor: input feature size resolved from `input.Shape[^1]` on first `Tensor{`; weights allocated then. |
| `RecurrentLayer(Int32,IVectorActivationFunction<>)` | Lazy ctor with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in this recurrent layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BroadcastBiases(Tensor<>,Int32)` | Broadcasts biases across the batch dimension. |
| `ClearGradients` | Resets the internal state of the recurrent layer. |
| `EnsureInitialized` | Allocates input/hidden weight tensors and biases using the resolved `_inputSize` and `_hiddenSize`. |
| `Forward(Tensor<>)` | Performs the forward pass of the recurrent layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors. |
| `GetMetadata` |  |
| `GetParameterGradients` | Gets all parameter gradients of the recurrent layer as a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the recurrent layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the recurrent layer with proper scaling. |
| `OnFirstForward(Tensor<>)` | Resolves `_inputSize` from `input.Shape[^1]` and propagates the full shape into the layer's resolved input/output shapes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the recurrent layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the recurrent layer using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters on GPU using the configured optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Tensor storing the bias parameters for each hidden neuron. |
| `_biasesGradient` | Stores the gradients of the loss with respect to the bias parameters. |
| `_hiddenSize` | Hidden state size — fixed at construction. |
| `_hiddenWeights` | Tensor storing the weight parameters for connections between previous hidden state and current hidden state. |
| `_hiddenWeightsGradient` | Stores the gradients of the loss with respect to the hidden weight parameters. |
| `_inputSize` | Resolved input feature size. |
| `_inputWeights` | Tensor storing the weight parameters for connections between inputs and hidden neurons. |
| `_inputWeightsGradient` | Stores the gradients of the loss with respect to the input weight parameters. |
| `_isInitialized` | True once weight tensors are allocated. |
| `_lastInput` | Stores the input tensor from the most recent forward pass for use in backpropagation. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass for use in backpropagation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

