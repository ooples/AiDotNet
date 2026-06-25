---
title: "MixtureOfMemoriesLayer<T>"
description: "Implements the Mixture of Memories (MoM) layer from Chou et al., 2025."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Mixture of Memories (MoM) layer from Chou et al., 2025.

## For Beginners

Think of this like having multiple filing cabinets (memories) instead of one:

Standard linear attention = one filing cabinet where all documents go.
Over time, the cabinet gets cluttered and finding specific documents is hard.

MoM = multiple filing cabinets, each for different topics:

- A router (like a librarian) decides which cabinet to file each new document in (write routing)
- When you need information, the librarian checks relevant cabinets and combines results (read routing)
- Each cabinet has its own retention policy: some keep documents forever, others regularly clean out (forget gate)

This prevents unrelated information from interfering with each other, which is the main weakness
of standard linear attention. The model learns to organize information across memories, much like
a well-organized library system.

The number of memories M is a key hyperparameter:

- More memories = less interference, more capacity, but more parameters
- Fewer memories = simpler model, but more compression needed
- 4-8 memories is typically a good balance

## How It Works

Standard linear attention maintains a single key-value memory state S. As the sequence grows,
all information is compressed into this one matrix, leading to interference: unrelated key-value
associations overwrite each other. MoM addresses this by maintaining **multiple independent memory
states** (S_1, S_2, ..., S_M) and using a learned router to selectively read from and write to them.

The architecture at each timestep t:

The routing mechanism is the key innovation: by directing different tokens to different memories,
MoM prevents interference between unrelated information. This is analogous to how Mixture of Experts
(MoE) routes tokens to different expert networks, but applied to the memory states of a linear
attention model.

The forget gate per memory allows selective retention: some memories can maintain long-term state
(high g) while others are more transient (low g), naturally specializing into different timescales.

**Reference:** Chou et al., "MoM: Mixture of Memories for Linear Sequence Modeling", 2025.
https://arxiv.org/abs/2502.13685

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixtureOfMemoriesLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Mixture of Memories (MoM) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `NumMemories` | Gets the number of memory states. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetReadRouterWeights` | Gets the read router weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWriteRouterWeights` | Gets the write router weights for external inspection. |
| `MoMForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | MoM forward: multi-memory state recurrence with routing. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SoftmaxBackward(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Backward pass for softmax: dLogits[i] = softmax[i] * (dOutput[i] - sum_j(softmax[j]*dOutput[j])) |
| `SoftmaxLastDim(Tensor<>,Int32,Int32,Int32)` | Computes softmax along the last dimension. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

