---
title: "LonghornLayer<T>"
description: "Implements the Longhorn layer from \"Longhorn: State Space Models are Amortized Online Learners\" (Liu et al., 2024)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Longhorn layer from "Longhorn: State Space Models are Amortized Online Learners" (Liu et al., 2024).

## For Beginners

Longhorn is a sequence model that maintains an internal "memory" which
works like a student learning from a stream of examples.

Imagine a student studying flashcards one at a time:

- Each flashcard has a "key" (the question) and a "value" (the answer)
- The student maintains a mental model (the state matrix S) that maps keys to values
- For each new flashcard, the student decides how much to "learn" from it (alpha)
- High alpha: "This is important, update my understanding significantly"
- Low alpha: "I already know this well, barely change my understanding"
- To answer a question (query), the student applies their mental model

The key difference from standard attention:

- Standard attention: Re-reads ALL flashcards every time (O(n^2) cost)
- Longhorn: Maintains a compressed summary that gets updated incrementally (O(n) cost)

This makes Longhorn much more efficient for long sequences while still capturing important patterns.

## How It Works

Longhorn reinterprets state space model state updates as amortized online learning. Rather than treating
the recurrent state as an opaque hidden state, Longhorn views it as a weight matrix of an online learner
that is continuously updated to predict values from keys. The decay/forget gate acts as a data-dependent
learning rate controlling how quickly the "learner" adapts to new observations.

The architecture:

The critical insight is that this state update rule is equivalent to an online learning rule:

- The state matrix S acts as the "model weights" of a linear predictor
- Each (k_t, v_t) pair is a new "training example"
- alpha_t is the "learning rate" that controls how much to update
- (1 - alpha_t) is the "retention rate" that controls how much to remember
- Querying S with q_t is equivalent to running inference on the online learner

Unlike GatedDeltaNet which uses a delta rule (error-corrective) update, Longhorn uses a simpler
exponential moving average update. This means it does not check what the state already knows before
writing -- it simply blends the old state with the new outer product. Despite this simplicity, the
online learning perspective enables principled initialization and understanding of the model's behavior.

**Reference:** Liu et al., "Longhorn: State Space Models are Amortized Online Learners", 2024.
https://arxiv.org/abs/2407.14207

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LonghornLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Longhorn layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head (modelDimension / numHeads). |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGroupNorm(Tensor<>,Int32,Int32)` | Applies group normalization to the recurrence output, normalizing within each head independently. |
| `CreateOnesLike(Tensor<>)` | Creates a tensor of ones with the same shape as the template tensor. |
| `Forward(Tensor<>)` |  |
| `GetAllTensors` | Returns all trainable parameter tensors in a consistent order for serialization. |
| `GetAlphaWeights` | Gets the alpha (learning rate gate) weights for external inspection or analysis. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection or analysis. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection or analysis. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GroupNormBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through group normalization, computing gradients for gamma, beta, and propagating gradients to the input. |
| `InitializeParameters` | Initializes all trainable parameters using Xavier/Glorot initialization for weight matrices and appropriate constants for biases. |
| `InitializeTensor2D(Tensor<>)` | Applies Xavier/Glorot uniform initialization to a 2D weight tensor. |
| `OnlineLearnerForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Longhorn online learner forward: exponential moving average of outer products. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

