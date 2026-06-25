---
title: "MegalodonLayer<T>"
description: "Implements the Megalodon layer from \"Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length\" (Ma et al., 2024, arXiv:2404.08801)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Megalodon layer from "Megalodon: Efficient LLM Pretraining and Inference with
Unlimited Context Length" (Ma et al., 2024, arXiv:2404.08801).

## For Beginners

Megalodon is like having a bank of tuning forks that each ring at
different frequencies.

Imagine you are analyzing a complex audio signal:

- A standard EMA is like a simple echo that fades over time -- it can only remember things

that happened recently, and the memory just gets quieter (exponential decay).

- Megalodon's CEMA is like a set of tuning forks. Each fork vibrates at its own frequency.

When you tap a tuning fork (give it input), it rings and slowly fades, but it OSCILLATES
while fading. This means it naturally picks up patterns that repeat at that frequency.

The "complex" in CEMA means each decay coefficient has two parts:

- A real part: controls how fast the oscillation fades (the damping)
- An imaginary part: controls how fast it oscillates (the frequency)

Together, many CEMA dimensions with different frequencies act like a Fourier-style analyzer
that decomposes the input into its frequency components, all within linear O(n) complexity.

The timestep normalization prevents the "tuning forks" from getting too loud over very long
sequences, which is why Megalodon can handle unlimited context length.

## How It Works

Megalodon extends the MEGA (Moving Average Equipped Gated Attention) architecture with two key
innovations that enable unlimited context length:

The CEMA mechanism is the core innovation. Standard EMA uses real-valued decay coefficients,
which can only model exponential decay. By making alpha complex-valued, the state evolves as
a damped oscillation: |alpha| controls the decay rate, while arg(alpha) controls the oscillation
frequency. This gives each EMA dimension a unique frequency response, enabling the model to
selectively attend to different periodicities in the input.

Timestep normalization is critical for unlimited context. Without it, the accumulated EMA state
grows with sequence length, causing numerical instability. By normalizing per-timestep, Megalodon
maintains stable representations regardless of context length.

**Reference:** Ma et al., "Megalodon: Efficient LLM Pretraining and Inference with Unlimited
Context Length", arXiv:2404.08801, 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MegalodonLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Megalodon layer with CEMA, timestep normalization, and gated attention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmaDimension` | Gets the EMA state dimension (number of complex EMA channels). |
| `HeadDimension` | Gets the dimension per attention head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AttentionBackward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through the causal multi-head attention. |
| `CEMAKernelBackward(Tensor<>,Int32,Int32)` | Backward pass through the CEMA recurrence. |
| `CEMAKernelForward(Tensor<>,Int32,Int32)` | Complex Exponential Moving Average forward pass with timestep normalization. |
| `Forward(Tensor<>)` |  |
| `GetEmaAlphaImag` | Gets the CEMA alpha coefficients (imaginary part) for external inspection. |
| `GetEmaAlphaReal` | Gets the CEMA alpha coefficients (real part) for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `MultiHeadAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Multi-head causal attention forward pass with scaled dot-product. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `TimestepNormBackward(Tensor<>,Int32,Int32)` | Backward pass through timestep normalization. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

