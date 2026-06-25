---
title: "DenseLayer<T>"
description: "Represents a fully connected (dense) layer in a neural network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected (dense) layer in a neural network.

## For Beginners

A dense layer is like a voting system where every input gets to vote on every output.

Think of it like this:

- Each input sends information to every output
- Each connection has a different "importance" (weight)
- The layer learns which connections should be strong and which should be weak

For example, in an image recognition task:

- One input might detect a curved edge
- Another might detect a straight line
- The dense layer combines these features to recognize higher-level patterns

Dense layers are the building blocks of many neural networks because they can learn
almost any relationship between inputs and outputs, given enough neurons and training data.

## How It Works

A dense layer connects every input neuron to every output neuron, with each connection having
a learnable weight. This is the most basic and widely used type of neural network layer.
Dense layers are capable of learning complex patterns by adjusting these weights during training.

**Thread Safety:** This layer is not thread-safe. Each layer instance maintains internal state
during forward and backward passes. If you need concurrent execution, use separate layer instances
per thread or synchronize access to shared instances.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenseLayer(Int32,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `DenseLayer` class with the specified input and output sizes and a scalar activation function. |
| `DenseLayer(Int32,IVectorActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `DenseLayer` class with the specified input and output sizes and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the regularization auxiliary loss. |
| `IsInitialized` |  |
| `L1Strength` | Gets or sets the L1 regularization strength (used when Regularization is L1 or L1L2). |
| `L2Strength` | Gets or sets the L2 regularization strength (used when Regularization is L2 or L1L2). |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `Regularization` | Gets or sets the type of regularization to apply. |
| `SupportsGpuExecution` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (weight regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivationAutodiff(ComputationNode<>)` | Applies activation function using autodiff operations. |
| `ClearGradients` | Clears stored gradients for weights and biases. |
| `Clone` | Creates a deep copy of the layer with the same configuration and parameters. |
| `ComputeActivationGradientGpu(DirectGpuTensorEngine,Tensor<>)` | Computes activation gradient using GPU-resident backward operations. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for weight regularization (L1, L2, or both). |
| `ConvertToOnnx(OnnxGraphBuilder,OnnxLayerInputs)` | Emits this Dense layer as an ONNX `Gemm` node (Y = A·B + C) with the layer's weights and biases as initializers. |
| `Dispose(Boolean)` | Releases resources used by this layer, including GPU tensor handles. |
| `EnsureInitialized` | Ensures that weights are allocated and initialized for lazy initialization. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass, keeping tensors on GPU. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the weight regularization auxiliary loss. |
| `GetBiases` | Gets the biases tensor of the layer. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Gets the gradients of all trainable parameters in this layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the weights tensor of the layer. |
| `InitializeParameters` | Initializes the weights and biases with appropriate values. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size from input.Shape[^1] on first forward. |
| `ResetState` | Resets the internal state of the layer. |
| `ResolveDefaultInitKind` | Single resolver that maps the layer's current activation function to the appropriate init strategy family. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SetWeights(Tensor<>)` | Sets the weights of the layer to specified values. |
| `ShapeInferenceOutput(Tensor<>)` | Processes the input data through the dense layer. |
| `UpdateParameters()` | Updates the layer's parameters (weights and biases) using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias values added to each output neuron. |
| `_biasesGradient` | Temporary storage for bias gradients during backpropagation. |
| `_isInitialized` | Tracks whether lazy initialization has been completed. |
| `_lastInput` | Stored input data from the most recent forward pass, used for backpropagation. |
| `_originalInputShape` | The original shape of the input tensor, used to restore shape after forward pass. |
| `_weights` | The weight matrix that connects input neurons to output neurons. |
| `_weightsGradient` | Temporary storage for weight gradients during backpropagation. |
| `_weightsHalf` | fp16-resident copy of the weight matrix, used only when `LowPrecisionResident` is set (foundation-scale inference). |

