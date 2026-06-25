---
title: "SpikingLayer<T>"
description: "Represents a layer of spiking neurons that model the biological dynamics of neural activity."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer of spiking neurons that model the biological dynamics of neural activity.

## For Beginners

This layer mimics how real neurons in the brain work, using "spikes" instead of smooth values.

Think of each neuron as a tiny battery that:

- Builds up electrical charge over time (membrane potential)
- Fires a signal (spike) when the charge reaches a threshold
- Needs a short recovery period after firing (refractory period)
- Has different ways of charging and discharging (different neuron models)

Benefits include:

- More biologically realistic modeling of neural activity
- Potential for energy-efficient computation (spikes are sparse)
- Ability to process time-dependent information naturally

For example, spiking neurons can directly model the timing patterns in speech or detect motion in video
by responding to when things change rather than constantly processing every value.

## How It Works

A spiking layer implements various biologically-inspired neuron models that operate with discrete spike events
rather than continuous activation values. The layer supports several neuron types including Leaky Integrate-and-Fire,
Izhikevich, Adaptive Exponential, and Hodgkin-Huxley models. Spiking neurons are characterized by their membrane
potential dynamics, threshold-crossing spike generation, and refractory periods after firing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpikingLayer(Int32,Int32,SpikingNeuronType,Double,Double)` | Initializes a new instance of the `SpikingLayer` class with the specified dimensions and neuron type. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Deserializes the layer's parameters and state from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of the spiking layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-accelerated forward pass for spiking neurons. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ProcessSpikes(Tensor<>)` | Processes input through the selected neuron model to generate spikes. |
| `ResetState` | Resets the internal state of the spiking layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Serializes the layer's parameters and state to a binary stream. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateAdaptiveExponential(Tensor<>)` | Updates the state of Adaptive Exponential Integrate-and-Fire neurons. |
| `UpdateHodgkinHuxley(Tensor<>)` | Updates the state of Hodgkin-Huxley neurons. |
| `UpdateIntegrateAndFire(Tensor<>)` | Updates the state of Integrate-and-Fire neurons. |
| `UpdateIzhikevich(Tensor<>)` | Updates the state of Izhikevich neurons. |
| `UpdateLeakyIntegrateAndFire(Tensor<>)` | Updates the state of Leaky Integrate-and-Fire neurons. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients and learning rate. |
| `UpdateParameters(Vector<>)` | Updates the layer parameters with new values. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_a` | Time scale of recovery variable in Izhikevich model. |
| `_a_adex` | Subthreshold adaptation in Adaptive Exponential model. |
| `_adaptationVariable` | Adaptation variable for Adaptive Exponential neuron model. |
| `_b` | Sensitivity of recovery variable to membrane potential in Izhikevich model. |
| `_b_adex` | Spike-triggered adaptation in Adaptive Exponential model. |
| `_bias` | Bias values for output neurons. |
| `_biasGradients` | Accumulated gradients for bias updates during training. |
| `_c` | After-spike reset value of membrane potential in Izhikevich model. |
| `_d` | After-spike reset of recovery variable in Izhikevich model. |
| `_deltaT` | Sharpness of exponential term in Adaptive Exponential model. |
| `_hGate` | Sodium inactivation gating variable for Hodgkin-Huxley model. |
| `_lastInput` | Stores the input tensor from the most recent forward pass. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass. |
| `_mGate` | Sodium activation gating variable for Hodgkin-Huxley model. |
| `_membranePotential` | Current membrane potential for each output neuron. |
| `_nGate` | Potassium activation gating variable for Hodgkin-Huxley model. |
| `_neuronType` | The type of spiking neuron model to use. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_recoveryVariable` | Recovery variable for Izhikevich neuron model. |
| `_refractoryCountdown` | Countdown timer for refractory period for each output neuron. |
| `_refractoryPeriod` | Refractory period in time steps during which the neuron cannot fire again after spiking. |
| `_spikes` | Output spikes for each output neuron (1.0 for spike, 0.0 for no spike). |
| `_tau` | Time constant for membrane potential decay. |
| `_tauw` | Adaptation time constant in Adaptive Exponential model. |
| `_threshold` | Firing threshold for spike generation. |
| `_vT` | Threshold potential in Adaptive Exponential model. |
| `_weightGradients` | Accumulated gradients for weight updates during training. |
| `_weights` | Connection weights between input and output neurons. |

