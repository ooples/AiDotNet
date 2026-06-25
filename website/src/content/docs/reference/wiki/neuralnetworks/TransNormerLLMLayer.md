---
title: "TransNormerLLMLayer<T>"
description: "Implements the TransNormerLLM layer from \"TransNormerLLM: A Faster and Better LLM\" (Qin et al., 2023)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the TransNormerLLM layer from "TransNormerLLM: A Faster and Better LLM" (Qin et al., 2023).

## For Beginners

TransNormerLLM makes "linear attention" actually work well for language models.

Standard linear attention has a known problem: it tends to become numerically unstable during training
because the accumulated state matrix can grow without bound. TransNormerLLM fixes this with two tricks:

1. RMSNorm on Q and K: Before computing attention, the queries and keys are normalized. This is like

making sure all "questions" and "answers" have similar magnitude, preventing any single token from
dominating the accumulated state.

2. Exponential decay: Old information naturally fades away (controlled by gamma), preventing the state

from accumulating indefinitely. This is similar to RetNet's approach but simpler.

Together, these allow TransNormerLLM to match or exceed Transformer quality while being much faster
for long sequences (linear vs quadratic complexity).

## How It Works

TransNormerLLM uses "Lightning Attention" -- a linear attention variant with exponential decay and
efficient normalization. Unlike standard Transformers that use softmax attention (O(n^2)), Lightning
Attention achieves linear complexity O(n) by combining linear attention with a decay factor and
RMSNorm-based normalization.

The architecture:

The key innovations over standard linear attention:

- RMSNorm on Q and K prevents the magnitude explosion that plagues linear attention
- Exponential decay gamma provides a soft causal bias (like RetNet) without rotary PE
- Per-output RMSNorm stabilizes the attention output, preventing training instability

These simple modifications make linear attention competitive with softmax attention for LLMs.

**Reference:** Qin et al., "TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer", 2023.
https://arxiv.org/abs/2307.14995

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransNormerLLMLayer(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new TransNormerLLM layer with lightning attention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecayRate` | Gets the decay rate. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRMSNorm(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Applies RMSNorm per head dimension and returns the inverse RMS for backward. |
| `Forward(Tensor<>)` |  |
| `GetDecayRates` | Gets the per-head decay rates (gammas) for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `LightningAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Lightning attention forward pass: linear attention with exponential decay. |
| `RMSNormBackward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through RMSNorm. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

