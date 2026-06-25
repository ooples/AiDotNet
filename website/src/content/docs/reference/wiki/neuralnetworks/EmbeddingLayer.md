---
title: "EmbeddingLayer<T>"
description: "Represents an embedding layer that converts discrete token indices into dense vector representations."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an embedding layer that converts discrete token indices into dense vector representations.

## For Beginners

An embedding layer turns words or other symbols into lists of numbers that capture their meaning.

Imagine you have a dictionary where:

- Each word has an ID number (like "cat" = 5, "dog" = 10)
- The embedding layer gives each ID a unique "coordinate" in a multi-dimensional space
- Words with similar meanings end up with similar coordinates

For example:

- "Cat" might become [0.2, -0.5, 0.1, 0.8]
- "Kitten" might become [0.25, -0.4, 0.15, 0.7]
- "Computer" might become [-0.8, 0.2, 0.5, -0.3]

The embedding layer learns these representations during training, so that:

- Similar words end up close to each other
- Related concepts form clusters
- The vectors capture meaningful semantic relationships

This allows neural networks to work with text and other discrete tokens in a way
that captures their meaning and relationships.

## How It Works

An embedding layer maps discrete tokens (represented as indices) to continuous vector representations.
This is particularly useful for natural language processing tasks where words or tokens need to be
represented as dense vectors that capture semantic relationships. Each token is assigned a unique
vector in a high-dimensional space, allowing the model to learn meaningful representations.

**Thread Safety:** This layer is not thread-safe. Each layer instance maintains internal state
during forward and backward passes. If you need concurrent execution, use separate layer instances
per thread or synchronize access to shared instances.

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for embedding regularization. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `ScaleBySqrtDimension` | When `true`, the token-embedding lookup output is multiplied by `sqrt(embeddingDimension)` (Vaswani et al. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer can execute on GPU. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (embedding regularization) during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the EmbeddingLayer, which is embedding regularization. |
| `EnsureEmbeddingInitialized` | Materializes the embedding tensor on first access. |
| `Forward(Tensor<>)` | Performs the forward pass of the embedding layer, converting token indices to vector representations. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of the embedding layer on GPU. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the embedding regularization. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetMetadata` | Returns layer-specific metadata for serialization. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTokenEmbeddings(IReadOnlyList<Int32>)` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the embedding tensor with small random values. |
| `InitializeProjectionWeights(Tensor<>,Int32,Int32)` | Lazy-initializes the continuous-input projection weights with Xavier scaling. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the embedding matrix using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingGradient` | The gradients for the embedding tensor, computed during backpropagation. |
| `_embeddingTensor` | The embedding tensor that stores vector representations for each token in the vocabulary. |
| `_lastEmbeddingRegularizationLoss` | Stores the last computed embedding regularization loss for diagnostics. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_originalInputShape` | The original input shape, saved for backward pass. |
| `_projectionWeights` | Projection weights for continuous input. |
| `_vocabularySize` | Cached vocabulary size and embedding dimension. |

