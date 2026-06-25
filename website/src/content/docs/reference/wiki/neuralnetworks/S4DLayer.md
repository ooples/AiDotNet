---
title: "S4DLayer<T>"
description: "Implements a Diagonal State Space (S4D) layer from Gu et al., 2022."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements a Diagonal State Space (S4D) layer from Gu et al., 2022.

## For Beginners

Think of S4D as a set of independent oscillators, each tuned to a
different frequency. When you feed in a sequence, each oscillator responds to the parts of the signal
at its frequency, building up a rich representation of the input over time.

The key insight is that diagonal state matrices are much simpler than full matrices:

- Full S4: A is N x N -> complex eigenvalue decomposition needed
- S4D: A is diagonal -> each state dimension is just one number

This makes S4D much faster while being nearly as expressive. It's the foundation that led to
more advanced models like Mamba.

## How It Works

S4D simplifies the original S4 model by using a diagonal state matrix A, which greatly reduces
computational complexity while maintaining competitive performance. The diagonal structure means
each state dimension evolves independently, enabling efficient parallelization.

The layer implements the continuous-time state space model:

where A is diagonal (and typically complex-valued, stored as real/imaginary pairs for generic compatibility).
Discretization uses the Zero-Order Hold (ZOH) method:

During training, the layer supports a global convolution mode that convolves the entire sequence
at once using the closed-form convolution kernel. During inference, it uses an efficient recurrent
mode that processes one step at a time with O(1) per-step cost.

The A matrix is initialized using HiPPO-LegS (Legendre polynomials) from Gu et al., 2020,
which provides a mathematically principled initialization that captures long-range dependencies.
S4D-Lin uses A_n = -1/2 + ni for state dimension n, giving logarithmically-spaced frequencies.

**Reference:** Gu et al., "On the Parameterization and Initialization of Diagonal State Space Models", 2022.
https://arxiv.org/abs/2206.11893

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `S4DLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new S4D (Diagonal State Space) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InnerDimension` | Gets the inner dimension used for the SSM computation. |
| `ModelDimension` | Gets the model dimension (d_model) of this S4D layer. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `StateDimension` | Gets the SSM state dimension (N) controlling the number of independent oscillators. |
| `SupportsTraining` | Training is not yet supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBBar(Double,Double,Double,Double,Double)` | Compute B_bar = (exp(dt*A) - I) / A * B for given complex A and B. |
| `Forward(Tensor<>)` |  |
| `GetAImag` | Gets the A parameter (imaginary part) for external inspection. |
| `GetAReal` | Gets the A parameter (real part) for external inspection. |
| `GetDParameter` | Gets the D skip connection parameter for external inspection. |
| `GetLogDelta` | Gets the log-delta (discretization step) parameter for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `KernelBasedBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through the complex recurrent scan. |
| `KernelBasedForward(Tensor<>,Int32,Int32)` | Performs the S4D recurrent scan with complex-valued state transitions. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `RunSingleStateRecurrence(Tensor<>,Int32,Int32,Int32,Int32,Double,Double,Double,Double,Double,Double,Double)` | Runs the recurrence for a single (d, n) state dimension with given A, B, C parameters. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

