---
title: "LinearRecurrentUnitLayer<T>"
description: "Implements the Linear Recurrent Unit (LRU) layer from Orvieto et al., 2023."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Linear Recurrent Unit (LRU) layer from Orvieto et al., 2023.

## For Beginners

LRU is one of the simplest state space models that actually works well.

Think of it like a set of independent "memory cells", each vibrating at its own frequency:

- Each cell has a complex number (think: a spinning arrow) that controls how it decays and oscillates
- At each time step, the cell's state is multiplied by its complex eigenvalue (shrinks and rotates)

and then the new input is added in

- The output combines all these spinning memories back into real values

The critical trick is how the eigenvalues are parameterized:

- exp(-exp(nu)) ensures the arrow ALWAYS shrinks (stable)
- exp(theta) controls the rotation speed (frequency)
- The model learns which frequencies and decay rates are useful for the task

This is like having a bank of tunable resonators that can remember patterns at different
time scales, from very short (fast decay) to very long (slow decay, eigenvalue near 1).

Despite its simplicity, LRU matches or beats much more complex architectures on many
long-range benchmarks, showing that the parameterization matters more than the complexity
of the recurrence.

## How It Works

The Linear Recurrent Unit is a simple linear recurrence with diagonal complex-valued state
transitions that achieves competitive performance with more complex SSMs on long-range tasks.
It is the core recurrence used in Google's Griffin/Hawk architectures.

The architecture:

The key insight of LRU is that diagonal linear recurrences are surprisingly powerful when
properly parameterized. The exp(-exp(nu)) parameterization for eigenvalue magnitudes ensures:

- Stability: all eigenvalues are strictly inside the unit circle
- Expressivity: the model can learn eigenvalues very close to 1 (long memory) or close to 0 (short memory)
- Smooth gradients: the double-exponential parameterization avoids gradient issues at the boundary

Since the state matrix is diagonal, each state dimension evolves independently, enabling
O(n) parallel scan computation for the entire sequence. This is in contrast to full-matrix
recurrences which require O(n * d^2) per step.

Complex values are represented as pairs of real numbers throughout, since the generic type T
is real-valued. Each complex state dimension uses two real numbers (real part + imaginary part),
and complex multiplication is performed using the identity (a+bi)(c+di) = (ac-bd) + (ad+bc)i.

**Reference:** Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences", 2023.
https://arxiv.org/abs/2303.06349

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearRecurrentUnitLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Linear Recurrent Unit (LRU) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model) of this LRU layer. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `StateDimension` | Gets the state dimension (N) controlling the number of independent complex recurrence channels. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiagonalComplexRecurrence(Tensor<>,Int32,Int32)` | Performs the LRU diagonal recurrence with complex-valued state transitions. |
| `DiagonalComplexRecurrenceBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through the diagonal complex recurrence. |
| `Forward(Tensor<>)` |  |
| `GetDParameter` | Gets the D skip connection parameter for external inspection. |
| `GetNuParameter` | Gets the nu (log magnitude) parameter for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetThetaParameter` | Gets the theta (phase) parameter for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

