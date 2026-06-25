---
title: "HyenaLayer<T>"
description: "Implements the Hyena layer from \"Hyena Hierarchy: Towards Larger Convolutional Language Models\" (Poli et al., 2023, arXiv:2302.10866)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Hyena layer from "Hyena Hierarchy: Towards Larger Convolutional Language Models"
(Poli et al., 2023, arXiv:2302.10866).

## For Beginners

Hyena is an alternative to the attention mechanism used in Transformers.

In a standard Transformer, every token "looks at" every other token via attention, which costs
O(N^2) time for a sequence of length N. Hyena achieves a similar effect more efficiently:

- Instead of attention, it uses **long convolutions** that slide a filter across the entire

sequence. Think of this like a sliding window that can "see" all positions.

- The convolution filters are not stored as giant arrays. Instead, a tiny neural network generates

the filter values on the fly from position numbers. This is called an "implicit filter."

- Between convolution steps, **data-dependent gates** (element-wise multiplications with

projections of the input) allow the model to selectively amplify or suppress information,
similar to how attention selectively focuses on relevant tokens.

- Stacking multiple rounds of "gate then convolve" (controlled by the `order` parameter)

gives the model enough expressivity to rival attention.

The result: Hyena can process much longer sequences than standard Transformers, because its
cost grows as O(N log N) instead of O(N^2).

## How It Works

Hyena replaces the standard attention mechanism with a hierarchy of long implicit convolutions
gated by data-dependent projections. This achieves sub-quadratic O(N log N) complexity while
matching or approaching Transformer quality on many sequence modeling benchmarks.

The architecture works as follows:

The long convolutions are the key innovation. Instead of storing an explicit kernel of length L
(which would be expensive for long sequences), Hyena uses a small MLP that takes a positional
encoding t/L as input and outputs the filter value h(t). This "implicit parameterization" means
the filter can span the entire sequence length with very few parameters. The convolution itself
can be computed efficiently in O(N log N) via FFT, though this implementation uses time-domain
convolution for clarity and correctness.

**Reference:** Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models", 2023.
``

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyenaLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Hyena layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FilterDim` | Gets the hidden dimension of the implicit filter network. |
| `ModelDimension` | Gets the model dimension. |
| `Order` | Gets the Hyena order (number of gated convolution stages). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SequenceLength` | Gets the sequence length this layer was configured for. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CausalConvolution(Tensor<>,Tensor<>,Int32,Int32)` | Performs causal (left-padded) convolution in the time domain. |
| `ClearGradients` |  |
| `ComputeSiLUDerivative(Tensor<>)` | Computes the derivative of the SiLU (Swish) activation function. |
| `Forward(Tensor<>)` |  |
| `GetInputProjectionWeights(Int32)` | Gets the input projection weights for a given index (0 = value, 1..N = gates). |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

