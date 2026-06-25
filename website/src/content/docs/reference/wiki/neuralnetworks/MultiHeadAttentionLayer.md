---
title: "MultiHeadAttentionLayer<T>"
description: "Implements a multi-head attention layer for neural networks, a key component in transformer architectures."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a multi-head attention layer for neural networks, a key component in transformer architectures.

## For Beginners

Multi-head attention is like having multiple "experts" look at the same information
from different perspectives. Each "head" focuses on different parts of the input, allowing the model
to capture various relationships in the data simultaneously. This is similar to how you might ask
several friends for advice on a decision - each person might notice different important factors.

## How It Works

**Thread Safety:** This layer is not thread-safe. Each layer instance maintains internal state
during forward and backward passes. If you need concurrent execution, use separate layer instances
per thread or synchronize access to shared instances.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiHeadAttentionLayer(Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new multi-head attention layer with the specified dimensions and head count. |
| `MultiHeadAttentionLayer(Int32,Int32,IVectorActivationFunction<>,IInitializationStrategy<>)` | Creates a new multi-head attention layer with the specified dimensions and head count. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the attention entropy auxiliary loss. |
| `HeadCount` | Gets the number of attention heads in this layer. |
| `HeadDiversityWeight` | Gets or sets the weight for head diversity penalty. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `PositionalEncoding` | Gets the positional encoding type used by this attention layer. |
| `RoPETheta` | Gets the RoPE theta parameter if RoPE is configured, or the default 10000.0. |
| `SupportsGpuExecution` | Indicates whether this layer supports GPU-resident execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (attention regularization) should be used during training. |
| `UseCausalMask` | Gets or sets whether causal masking is applied during attention computation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for attention regularization (entropy + head diversity). |
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two tensors. |
| `ConfigurePositionalEncoding(PositionalEncodingType,Double,Int32)` | Configures positional encoding for this attention layer. |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `EnsureWeightsAllocated` | Ensures the Q/K/V/O/bias tensors are allocated and populated. |
| `FillTensorRandomScaled(Tensor<>,SimdRandom,)` | Fills a tensor with scaled random values in [-0.5, 0.5] * scale using SimdRandom. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Performs the forward pass of the multi-head attention layer. |
| `ForwardGpu(Tensor<>[])` | GPU-resident forward pass for multi-head attention. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the attention regularization auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetKeyWeights` | Gets the key projection weights tensor for JIT compilation. |
| `GetMetadata` | Returns layer-specific metadata required for cloning/serialization. |
| `GetOutputWeights` | Gets the output projection weights tensor for JIT compilation. |
| `GetParameterGradients` | Resets the internal state of the multi-head attention layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Extracts all parameters (weights and biases) from the layer into a single vector. |
| `GetQueryWeights` | Gets the query projection weights tensor for JIT compilation. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetValueWeights` | Gets the value projection weights tensor for JIT compilation. |
| `InitializeParameters` | Initializes the weights and biases of the layer. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; passthrough since output equals input shape. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all parameters (weights and biases) of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SetTrainingMode(Boolean)` | Overrides the base SetTrainingMode to drop training-only caches when transitioning to eval mode. |
| `TryDeclareShape` | AiDotNet#1370 shape oracle override: MultiHeadAttentionLayer's constructor takes `(headCount, headDimension)` from which the embedding dim `= headCount * headDimension` is fully determined, and the four projection weight matrices (Q / K / V… |
| `TryFusedAttentionInference(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Attempts the single fused multi-head-attention inference kernel (`IEngine.MultiHeadAttentionForward`) for a pure self-attention block. |
| `UpdateParameters()` | Updates the layer's parameters (weights and biases) using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | Cached embedding dimension so `EnsureInitialized` knows what shape to allocate for the weight tensors when taking the lazy path. |
| `_headCount` | The number of attention heads in this layer. |
| `_headDimension` | The size of each attention head. |
| `_inputPortsCache` | Declares named input ports for this multi-input layer. |
| `_isInitialized` | True once `EnsureInitialized` has allocated and populated the weight/bias tensors. |
| `_keyWeights` | Tensor of weights for transforming input into key representations. |
| `_keyWeightsGradient` | Tensor storing gradients for key weights calculated during backward pass. |
| `_lastAttentionContext` | Cached attention context (pre-projection input) for computing output weights gradient. |
| `_lastAttentionScores` | Cached attention scores from the forward pass for use in the backward pass. |
| `_lastInput` | Cached input from the forward pass for use in the backward pass. |
| `_lastOutput` | Cached output from the forward pass for use in the backward pass. |
| `_lastPreActivationOutput` | Cached pre-activation output (before activation function) for computing activation derivative correctly. |
| `_outputBias` | Tensor of biases added to the final output. |
| `_outputBiasGradient` | Tensor storing gradients for output bias calculated during backward pass. |
| `_outputWeights` | Tensor of weights for the final output projection. |
| `_outputWeightsGradient` | Tensor storing gradients for output weights calculated during backward pass. |
| `_queryWeights` | Tensor of weights for transforming input into query representations. |
| `_queryWeightsGradient` | Tensor storing gradients for query weights calculated during backward pass. |
| `_valueWeights` | Tensor of weights for transforming input into value representations. |
| `_valueWeightsGradient` | Tensor storing gradients for value weights calculated during backward pass. |

