---
title: "RodimusLayer<T>"
description: "Implements the Rodimus layer from \"Rodimus: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions\" (He et al., 2025)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Rodimus layer from "Rodimus: Breaking the Accuracy-Efficiency Trade-Off
with Efficient Attentions" (He et al., 2025).

## For Beginners

Rodimus is a smart attention mechanism that can adjust its "focus level"
depending on the input.

Think of reading a textbook:

- Sometimes you need to focus sharply on one specific definition (low temperature = very selective)
- Other times you need to understand the general theme of a paragraph (high temperature = broad focus)
- A good reader adjusts their focus level based on what they're reading

Rodimus does exactly this:

- It learns a "temperature" for each position that controls focus sharpness
- Low temperature = laser focus on the most relevant information
- High temperature = broad survey of all available information
- The temperature is "data-dependent" meaning it adjusts based on the input itself

Combined with a gated recurrence (like LSTM but for a matrix-valued state), this gives
Rodimus the quality of Transformer attention with the efficiency of linear recurrence.
The gated recurrence maintains a running state matrix that gets selectively updated
at each step, avoiding the O(n^2) cost of full attention.

## How It Works

Rodimus combines a data-dependent tempered selection mechanism with gated linear recurrence
to achieve both high quality and linear-time efficiency. The key innovation is using a
temperature-scaled softmax for selective state updates, allowing the model to dynamically
control the sharpness of its attention/selection mechanism.

The architecture:

The temperature parameter tau is crucial: it controls the "sharpness" of attention.

- Low temperature (tau near 0): Very selective, focuses on the best-matching key (like argmax)
- High temperature (tau near infinity): Uniform attention, treats all keys equally
- Data-dependent: The model learns when to be selective vs. when to spread attention

This allows Rodimus to adaptively decide: "Should I focus sharply on one specific key-value pair
(low temp), or should I aggregate broadly (high temp)?" This breaks the typical accuracy-efficiency
trade-off because the model can be precise when needed and efficient otherwise.

**Reference:** He et al., "Rodimus: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions", 2025.
https://arxiv.org/abs/2410.06577

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RodimusLayer(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Rodimus layer with data-dependent tempered selection. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseTemperature` | Gets the base temperature value. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
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
| `GetTemperatureWeights` | Gets the temperature weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `Softplus()` | Computes softplus activation: softplus(x) = ln(1 + exp(x)). |
| `SoftplusDerivative()` | Computes the derivative of softplus: softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x)). |
| `TemperedRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Tempered selection with gated linear recurrence. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

