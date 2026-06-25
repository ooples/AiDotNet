---
title: "Mamba2Block<T>"
description: "Implements the Mamba-2 block using the State Space Duality (SSD) framework from Dao and Gu, 2024."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Mamba-2 block using the State Space Duality (SSD) framework from Dao and Gu, 2024.

## For Beginners

Mamba-2 is the faster sequel to Mamba.

Think of it this way:

- Mamba-1 processes each token one at a time (like reading a book word by word)
- Mamba-2 processes chunks of tokens at once (like reading a paragraph at a time)

Within each chunk, it uses a matrix multiplication (like attention) that is very fast on GPUs.
Between chunks, it uses the efficient state-passing from Mamba-1.
The result: same quality as Mamba-1, but 2-8x faster in practice.

The "multi-head" aspect is similar to multi-head attention in Transformers:
each head can focus on different patterns independently.

## How It Works

Mamba-2 reveals a deep connection between state space models and structured attention:
the selective scan can be expressed as multiplication by a semi-separable matrix, which is
equivalent to a form of structured masked attention. This duality enables 2-8x faster
computation than Mamba-1 through hardware-efficient block-wise parallel algorithms.

Key differences from Mamba-1:

- Multi-head SSM: state is partitioned into heads (like multi-head attention)
- Block-size chunking: sequence is processed in chunks for parallel within-chunk computation
- Simplified parameterization: A is scalar per head (not per-dimension), B and C are shared across heads
- SSD computation: combines efficient chunked quadratic attention within blocks with linear recurrence across blocks

The architecture per timestep:

**Reference:** Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", 2024.
https://arxiv.org/abs/2405.21060

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mamba2Block(Int32,Int32,Int32,Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Mamba-2 block with State Space Duality (SSD) computation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSize` | Gets the chunk size used for block-wise parallel computation. |
| `ConvKernelSize` | Gets the convolution kernel size. |
| `HeadDimension` | Gets the dimension per head. |
| `InnerDimension` | Gets the inner dimension (d_inner = modelDim * expandFactor). |
| `ModelDimension` | Gets the model dimension (d_model) of this Mamba-2 block. |
| `NumHeads` | Gets the number of SSM heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `StateDimension` | Gets the SSM state dimension (N). |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRMSNorm(Tensor<>,Int32,Int32)` | Applies RMS normalization to the SSD output. |
| `DepthwiseConv1DBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass for depthwise causal Conv1D using explicit per-element computation. |
| `DepthwiseConv1DForward(Tensor<>,Int32,Int32)` | Depthwise causal Conv1D forward using explicit per-element computation. |
| `Forward(Tensor<>)` |  |
| `GetALogParameter` | Gets the A_log parameter tensor for external inspection. |
| `GetDParameter` | Gets the D skip connection parameter for external inspection. |
| `GetInputProjectionWeights` | Gets the input projection weights for external inspection or quantization. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection or quantization. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ReduceSumAxes01(Tensor<>,Int32,Int32,Int32)` | Workaround for Engine.ReduceSum multi-axis [0,1] bug (AiDotNet.Tensors PR #62). |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SSDBackward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32,Tensor<>,Tensor<>,Tensor<>)` | Backward pass through the SSD multi-head selective scan. |
| `SSDForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | SSD (State Space Duality) forward computation using multi-head selective scan. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

