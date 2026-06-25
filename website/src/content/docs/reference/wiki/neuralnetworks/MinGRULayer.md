---
title: "MinGRULayer<T>"
description: "Implements the minGRU layer from \"Were RNNs All We Needed?\" (Feng et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the minGRU layer from "Were RNNs All We Needed?" (Feng et al., 2024).

## For Beginners

minGRU is a stripped-down version of the GRU that is much simpler yet
surprisingly powerful.

Imagine you are writing notes while listening to a lecture:

- At each moment, you decide how much of your old notes to keep vs. how much new content to write down.
- In a standard GRU, your decision depends on both the new content AND everything you have written so far.

This creates a chain of dependencies: step 1 must finish before step 2 can start.

- In minGRU, your decision depends ONLY on the new content. You can look at every slide in the lecture

in parallel, decide what is important, and then combine everything in one fast sweep.

This is like the difference between:

- Reading a book one page at a time, deciding what to remember based on what you read so far (slow, sequential)
- Skimming all pages at once, marking important parts, then combining them in a single organized pass (fast, parallel)

Despite being simpler, minGRU matches standard GRU and LSTM performance on most benchmarks,
showing that the hidden-state dependency in the gate was often unnecessary.

## How It Works

minGRU is a minimal variant of the Gated Recurrent Unit that removes the hidden-state dependency
from the gate computation. This seemingly small change has a profound consequence: the recurrence
becomes a **linear recurrence** in log-space, enabling efficient parallel training via prefix
sum (parallel scan) algorithms that run in O(log n) sequential steps instead of O(n).

The standard GRU equations:

The minGRU simplification:

**Why this enables parallel training:** Because z_t and h_tilde_t depend only on x_t (not h_{t-1}),
they can be precomputed for all timesteps in parallel. The recurrence h_t = (1-z_t)*h_{t-1} + z_t*h_tilde_t
is then a linear first-order recurrence of the form h_t = a_t*h_{t-1} + b_t where a_t = (1-z_t) and
b_t = z_t*h_tilde_t are known constants. Linear recurrences can be solved with parallel prefix sum
(also called parallel scan) in O(log n) parallel time, compared to O(n) for sequential RNNs.

In log-space, the recurrence becomes numerically stable:

which can be computed via the log-sum-exp trick in a parallel scan.

This implementation uses the sequential recurrence for correctness and clarity. For production
training on GPU, the parallel scan formulation would be used for O(log n) wall-clock time.

**Reference:** Feng et al., "Were RNNs All We Needed?", 2024.
https://arxiv.org/abs/2410.01201

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinGRULayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new minGRU layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpandedDimension` | Gets the expanded internal dimension used for the recurrence. |
| `ExpansionFactor` | Gets the expansion factor applied to the model dimension for the internal recurrence. |
| `ModelDimension` | Gets the model dimension (input/output width). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetGateWeights` | Gets the gate weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor2D(Tensor<>)` | Xavier/Glorot uniform initialization for 2D weight tensors. |
| `MinGRURecurrenceForward(Tensor<>,Tensor<>,Int32,Int32)` | Computes the minGRU recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

