---
title: "TTTLayer<T>"
description: "Implements the TTT (Test-Time Training) layer from Sun et al., 2024."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the TTT (Test-Time Training) layer from Sun et al., 2024.

## For Beginners

Traditional RNNs store their memory as a fixed-size vector (like a notepad
with limited space). TTT stores memory as a small neural network's weights (like having a student
who learns from each example).

At each step in the sequence:

- The "student" (inner model W) tries to predict the value from the key
- It computes how wrong it was (the loss)
- It takes a learning step to improve (gradient descent)
- Then it answers a query using its updated knowledge

This means TTT can adapt to the specific patterns in each sequence, much like how you get better
at a task the more examples you see. The inner learning rate (eta) controls how quickly the
student adapts -- too fast and it forgets old information, too slow and it cannot keep up.

TTT-Linear is competitive with Transformers and Mamba on language modeling benchmarks while
maintaining linear O(n) complexity in sequence length.

## How It Works

TTT replaces the fixed-size hidden state of traditional RNNs with a more expressive hidden state:
the WEIGHTS of a small inner model. At each timestep the layer performs a gradient-based learning
step on those weights, effectively training the inner model on the fly while processing the sequence.

This file implements the **TTT-Linear** variant where the inner model is a single linear map W.
The recurrence at each timestep t is:

The key insight is that the hidden state (W) grows in expressivity with the size of the inner model.
Unlike a fixed-size RNN state vector, W can represent arbitrary linear relationships and is updated
using a principled gradient descent rule (the delta rule). This makes TTT a bridge between
recurrent models and in-context learning: the model literally "learns at test time."

Multi-head operation: each head maintains its own inner weight matrix W_h of size [headDim, headDim].
This allows different heads to learn different relationships from the sequence, similar to multi-head
attention. The heads operate independently and their outputs are concatenated.

**Reference:** Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", 2024.
https://arxiv.org/abs/2407.04620

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TTTLayer(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new TTT (Test-Time Training) layer implementing the TTT-Linear variant. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `InnerLearningRate` | Gets the inner learning rate used for test-time training updates. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
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
| `LayerNormBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass for layer normalization. |
| `LayerNormForward(Tensor<>,Int32,Int32)` | Simple layer normalization across the last dimension. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `TTTLinearForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | TTT-Linear forward: inner model weight update via gradient descent on self-supervised loss. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |
| `two()` | Returns the constant 2.0 as type T. |

