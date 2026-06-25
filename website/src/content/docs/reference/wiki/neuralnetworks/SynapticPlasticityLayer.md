---
title: "SynapticPlasticityLayer<T>"
description: "Represents a synaptic plasticity layer that models biological learning mechanisms through spike-timing-dependent plasticity."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a synaptic plasticity layer that models biological learning mechanisms through spike-timing-dependent plasticity.

## For Beginners

This layer mimics how brain cells (neurons) learn by strengthening or weakening their connections.

Think of it like forming memories:

- When two connected neurons activate in sequence (one fires, then the other), their connection gets stronger
- When they activate in the opposite order, their connection gets weaker
- Over time, pathways that represent useful patterns become stronger

For example, imagine learning to associate a bell sound with food (like Pavlov's dog experiment):

- Initially, there's a weak connection between "hear bell" neurons and "expect food" neurons
- When the bell regularly comes before food, the connection strengthens
- Eventually, just the bell alone strongly activates the "expect food" response

This mimics how real brains learn patterns and form associations between related events.

## How It Works

A synaptic plasticity layer implements biologically-inspired learning rules that modify connection strengths based on 
the relative timing of pre- and post-synaptic neuron activations. This implements spike-timing-dependent plasticity (STDP),
a form of Hebbian learning observed in biological neural systems. The layer maintains traces of neuronal activity and
applies long-term potentiation (LTP) and long-term depression (LTD) based on the temporal relationship between spikes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SynapticPlasticityLayer(Int32,Double,Double,Double,Double,Double,Double)` | Initializes a new instance of the `SynapticPlasticityLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the synaptic plasticity layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_homeostasisRate` | The rate at which homeostatic mechanisms regulate synaptic strength. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastOutput` | The output tensor from the last forward pass. |
| `_maxWeight` | The maximum allowed value for a synaptic weight. |
| `_minWeight` | The minimum allowed value for a synaptic weight. |
| `_postsynapticSpikes` | The current spike state of postsynaptic neurons (binary). |
| `_postsynapticTraces` | The activity traces of postsynaptic neurons. |
| `_presynapticSpikes` | The current spike state of presynaptic neurons (binary). |
| `_presynapticTraces` | The activity traces of presynaptic neurons. |
| `_stdpLtdRate` | The rate at which long-term depression (weakening) occurs. |
| `_stdpLtpRate` | The rate at which long-term potentiation (strengthening) occurs. |
| `_traceDecay` | The decay rate of activity traces. |
| `_weights` | The weight matrix representing connection strengths between neurons. |

