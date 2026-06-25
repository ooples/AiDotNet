---
title: "S5Layer<T>"
description: "Implements the Simplified State Space (S5) layer from Smith et al., 2023."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Simplified State Space (S5) layer from Smith et al., 2023.

## For Beginners

S5 is an efficient sequence model that competes with Transformers
on long-range tasks while being much more computationally efficient.

Think of S5 as a bank of coupled oscillators processing a signal:

- Each oscillator has a complex frequency (how fast it vibrates) and a decay rate
- Unlike S4D where each feature has its own independent oscillator bank, S5 shares

one oscillator bank across ALL features -- this is the "MIMO" (multi-input multi-output) part

- The B matrix controls how each input feature excites the oscillators
- The C matrix controls how oscillator states are combined to produce each output feature
- The D matrix provides a direct skip connection from input to output

The key advantages of S5 over S4D:

1. MIMO formulation: Features interact through shared state, enabling richer representations
2. Parallel scan: The entire sequence is processed in O(L log L) instead of O(L) sequential steps
3. Simpler architecture: Fewer parameters than S4D with comparable or better performance

The "simplified" in S5 refers to replacing S4's complex NPLR (Normal Plus Low-Rank) structure
with a straightforward diagonalization, making the model much easier to implement and reason about
while maintaining strong empirical performance.

## How It Works

S5 is a multi-input multi-output (MIMO) state space model that uses a diagonalized state
matrix with parallel scan for efficient sequence processing. Unlike S4D, which applies
independent single-input single-output (SISO) SSMs per feature, S5 uses a single MIMO SSM
that couples all input dimensions through shared state dynamics.

The continuous-time MIMO SSM is:

where A is diagonalized via eigendecomposition: A = V * Lambda * V^{-1}, with Lambda containing
complex eigenvalues. This diagonalization decouples the state dimensions while retaining the
MIMO structure through B and C matrices that mix input/output dimensions with state dimensions.

Discretization uses the Zero-Order Hold (ZOH) method:

The discrete recurrence x_t = A_bar * x_{t-1} + B_bar * u_t is then computed efficiently
using a parallel associative scan in O(L log L) time, where L is the sequence length.

The A matrix is initialized using the HiPPO framework (High-order Polynomial Projection Operator)
which provides optimal polynomial approximations for continuous signal history. The S5 paper
uses the HiPPO-LegS (Legendre-Scaled) initialization, which gives the diagonal eigenvalues
Lambda_n = -1/2 + n*pi*i, placing them along a vertical line in the left half-plane with
logarithmically spaced frequencies. This ensures stable dynamics with a rich frequency spectrum
for capturing long-range dependencies.

**Reference:** Smith et al., "Simplified State Space Layers for Sequence Modeling", ICLR 2023.
https://arxiv.org/abs/2208.04933

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `S5Layer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new S5 (Simplified State Space) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (H), the width of input and output at each sequence position. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `StateDimension` | Gets the SSM state dimension (N), the number of diagonal complex eigenvalues. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBBarScalar(Double,Double,Double,Double,Double)` | Performs the S5 MIMO recurrence with diagonalized complex state transitions. |
| `Forward(Tensor<>)` |  |
| `GetAImag` | Gets the A parameter (imaginary part of diagonal eigenvalues) for external inspection. |
| `GetAReal` | Gets the A parameter (real part of diagonal eigenvalues) for external inspection. |
| `GetDParameter` | Gets the D skip connection parameter for external inspection. |
| `GetLogDelta` | Gets the log-delta (discretization step) parameter for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `KernelBasedMIMOBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through the MIMO complex recurrent scan. |
| `KernelBasedMIMOForward(Tensor<>,Int32,Int32)` | Kernel-based MIMO forward for S5. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

