---
title: "BinarySpikingActivation<T>"
description: "Implements the Binary Spiking activation function for neural networks, particularly for spiking neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Binary Spiking activation function for neural networks, particularly for spiking neural networks.

## For Beginners

Binary Spiking activation models how real neurons in your brain generate electrical pulses.

Unlike standard activation functions that output continuous values, Binary Spiking:

- Outputs only 1 (spike/fire) or 0 (no spike)
- Neurons "fire" only when their input exceeds a threshold
- After firing, neurons typically have a "refractory period" before they can fire again

This activation creates the discrete, all-or-nothing behavior of biological neurons:

- Input below threshold ? Output = 0 (neuron remains silent)
- Input at or above threshold ? Output = 1 (neuron fires a spike)

Common uses include:

- Spiking Neural Networks (SNNs)
- Neuromorphic computing systems
- Models that process temporal information
- Energy-efficient neural networks for specialized hardware

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BinarySpikingActivation` | Initializes a new instance of the Binary Spiking activation function with default parameters. |
| `BinarySpikingActivation(,,)` | Initializes a new instance of the Binary Spiking activation function with custom parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the approximate derivative of the Binary Spiking function for a single value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the Binary Spiking function for a tensor input. |
| `Derivative(Vector<>)` | Calculates the derivative (gradient) of the Binary Spiking function for a vector input. |
| `GetThreshold` | Gets the firing threshold value used by this activation function. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |
| `WithThreshold()` | Creates a new instance of the Binary Spiking activation with a different threshold. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_derivativeSlope` | The slope of the approximated derivative curve used during training. |
| `_derivativeWidth` | The width of the region around the threshold where the derivative is non-zero. |
| `_threshold` | The firing threshold for neurons. |

