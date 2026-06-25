---
title: "MesaNetLayer<T>"
description: "Implements the MesaNet layer from Grazzi et al., 2025."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the MesaNet layer from Grazzi et al., 2025.

## For Beginners

Imagine you have a student (the inner model W) who learns from examples.

TTT approach: After each example, the student takes a small step toward the right answer
(gradient descent). The step size (learning rate) is tricky to set -- too big and the student
overshoots, too small and learning is slow.

MesaNet approach: After each example, the student computes the BEST POSSIBLE answer given all
examples seen so far, with a gentle preference toward their initial knowledge (regularization).
There is no step size to tune -- the answer is mathematically optimal.

The Woodbury trick makes this efficient: instead of re-solving the entire problem from scratch
each time, MesaNet incrementally updates the solution as new examples arrive. This is like
updating a running average instead of recomputing from all data points.

## How It Works

MesaNet ("Locally Optimal Test-Time Training") improves upon TTT by replacing gradient descent
with a closed-form ridge regression update for the inner model weights. Instead of taking a noisy
gradient step at each timestep, MesaNet computes the **locally optimal** weight matrix W_t that
minimizes the reconstruction error plus a regularization term toward the initial weights W_0.

The optimization at each timestep t:

The Woodbury identity allows rank-1 updates to the inverse covariance matrix P, making each step
O(d^2) instead of O(d^3) for a full matrix inversion. This keeps the overall complexity linear
in sequence length while achieving a strictly better update than gradient descent.

Why this is better than TTT's gradient descent:

- TTT: W_t = W_{t-1} - eta * grad (approximate, depends on learning rate choice)
- MesaNet: W_t = optimal solution of ridge regression (exact, only depends on lambda)
- No learning rate sensitivity -- lambda is more stable and interpretable
- Converges in one step per observation rather than requiring multiple gradient steps

**Reference:** Grazzi et al., "MesaNet: Locally Optimal Test-Time Training", 2025.
https://arxiv.org/abs/2506.05233

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MesaNetLayer(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new MesaNet layer implementing locally optimal test-time training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `Regularization` | Gets the regularization strength (lambda). |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetInnerWeightsInit` | Gets the inner model initial weights for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `LayerNormForward(Tensor<>,Int32,Int32)` | Simple layer normalization across the last dimension. |
| `MesaForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Mesa forward: ridge regression with incremental Woodbury updates. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

