---
title: "TimeEmbeddingLayer<T>"
description: "Represents a time embedding layer that encodes timesteps using sinusoidal embeddings for diffusion models."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a time embedding layer that encodes timesteps using sinusoidal embeddings for diffusion models.

## For Beginners

In diffusion models, the network needs to know "what time step are we at?"

- At early timesteps (t near 0), images are clean and noise is minimal
- At late timesteps (t near T), images are mostly noise
- The network needs this information to know how much denoising to apply

This layer encodes the timestep number into a rich vector representation that:

1. Uses sine and cosine functions at different frequencies (sinusoidal encoding)
2. Passes through a small neural network (MLP) to learn task-specific representations
3. Gets injected into every ResNet block of the U-Net

The sinusoidal encoding is inspired by transformer positional encodings:

- Low frequencies capture coarse time information
- High frequencies capture fine-grained time details

## How It Works

The time embedding layer converts scalar timesteps into high-dimensional embeddings using sinusoidal
functions, similar to positional encodings in transformers. This embedding is then projected through
a small MLP to produce the final time conditioning vector used in diffusion U-Net blocks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeEmbeddingLayer(Int32,Int32,Int32)` | Initializes a new instance of the `TimeEmbeddingLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSinusoidalEmbedding(Tensor<>)` | Computes sinusoidal embedding for the given timesteps. |
| `Forward(Tensor<>)` | Performs the forward pass of the time embedding layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the layer from a vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SiLU()` | Applies the SiLU (Swish) activation function: x * sigmoid(x). |
| `SiLUDerivative()` | Computes the derivative of SiLU activation. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDim` | The dimension of the sinusoidal embedding before MLP projection. |
| `_lastHidden` | Cached intermediate output after first linear + activation. |
| `_lastInput` | Cached input timesteps from last forward pass. |
| `_lastSinusoidalEmbed` | Cached sinusoidal embedding from last forward pass. |
| `_linear1Bias` | First linear layer biases: [outputDim] |
| `_linear1BiasGradient` | Gradient for first linear layer biases. |
| `_linear1Weights` | First linear layer weights: [embeddingDim, outputDim] |
| `_linear1WeightsGradient` | Gradient for first linear layer weights. |
| `_linear2Bias` | Second linear layer biases: [outputDim] |
| `_linear2BiasGradient` | Gradient for second linear layer biases. |
| `_linear2Weights` | Second linear layer weights: [outputDim, outputDim] |
| `_linear2WeightsGradient` | Gradient for second linear layer weights. |
| `_maxTimestep` | The maximum timestep value for scaling embeddings. |
| `_outputDim` | The dimension of the output after MLP projection. |

