---
title: "ExpertLayer<T>"
description: "Represents an expert module in a Mixture-of-Experts architecture, containing a sequence of layers."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an expert module in a Mixture-of-Experts architecture, containing a sequence of layers.

## For Beginners

Think of an Expert as a mini neural network that specializes in a particular task.

In a Mixture-of-Experts system:

- You have multiple "experts" (mini-networks), each with their own layers
- Each expert learns to be good at handling certain types of inputs
- A routing mechanism decides which experts should process each input
- The final output combines the predictions from the selected experts

For example, in a language model:

- One expert might specialize in technical vocabulary
- Another might handle conversational language
- Another might focus on formal writing
- The router learns to send each input to the most appropriate expert(s)

This allows the model to scale to very large sizes while keeping computation efficient,
since only a subset of experts are activated for each input.

## How It Works

An Expert is a container for a sequence of neural network layers that are executed sequentially.
In a Mixture-of-Experts (MoE) architecture, multiple experts process the same input, and their outputs
are combined based on learned routing weights. Each expert can specialize in processing different
types of inputs or patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpertLayer(List<ILayer<>>,Int32[],Int32[],IActivationFunction<>)` | Initializes a new instance of the `ExpertLayer` class with the specified layers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters across all layers in this expert. |
| `SupportsGpuExecution` | Gets a value indicating whether this expert supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this expert supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this expert, including all contained layers. |
| `Forward(Tensor<>)` | Processes the input data through all layers in sequence. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors by chaining through all layers. |
| `GetActivationType` | Gets the FusedActivationType for the expert's activation function. |
| `GetParameterGradients` | Sets all trainable parameters in all layers from a single vector. |
| `GetParameters` | Gets all trainable parameters from all layers as a single vector. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of all layers, clearing any cached values. |
| `UpdateParameters()` | Updates all trainable parameters in all layers using the specified learning rate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastPreActivationOutput` | Stores the pre-activation output for use in backpropagation. |
| `_layers` | The sequence of layers that make up this expert. |

