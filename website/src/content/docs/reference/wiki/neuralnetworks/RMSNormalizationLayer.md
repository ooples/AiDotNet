---
title: "RMSNormalizationLayer<T>"
description: "Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019).

## For Beginners

RMSNorm is a simpler, faster cousin of LayerNorm.

LayerNorm: subtracts the mean, divides by standard deviation, then scales and shifts.
RMSNorm: skips the mean-subtraction step, divides by the root-mean-square, then scales.

The "skip the mean" part isn't a shortcut — Zhang & Sennrich showed it works
just as well in practice but costs less compute. Every modern LLM-style text
encoder (T5, LLaMA, Gemma, Qwen, ChatGLM) uses RMSNorm because it's both
paper-validated and faster.

## How It Works

RMSNorm rescales each sample's feature vector by its root-mean-square magnitude
(without re-centering on the mean), then multiplies by a learnable per-feature
gain γ. Concretely:

`RMSNorm(x)_i = (x_i / sqrt(mean(x²) + ε)) · γ_i`

This is the normalization used by T5 (Raffel et al. 2020), LLaMA (Touvron et al.
2023), Gemma (Gemma Team 2024), Qwen2 (Yang et al. 2024), and ChatGLM3
(Zeng et al. 2023). Unlike standard LayerNorm there is NO learnable shift β —
the paper-canonical RMSNorm formulation only includes a multiplicative gain.

**Reference:** Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RMSNormalizationLayer(Double)` | Initializes a new instance of the `RMSNormalizationLayer` class. |
| `RMSNormalizationLayer(Int32,Double)` | AiDotNet#1370 eager-init constructor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Forward(Tensor<>)` | Performs the forward pass of RMSNorm via composed Engine ops so the gradient tape records each step and Backward is supplied automatically by autodiff — no manual backward kernel needed. |
| `GetEpsilon` | Gets the epsilon value used for numerical stability. |
| `GetGamma` | Gets the gain (γ) parameters of the layer. |
| `GetGammaTensor` | Gets the gain (γ) tensor for JIT compilation and internal use. |
| `GetMetadata` | Returns layer-specific metadata required for cloning/serialization. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` | Resolves `featureSize` from `input.Shape[^1]` on the first forward call and allocates the γ tensor. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

