---
title: "UniversalDifferentialEquation<T>"
description: "Implements Universal Differential Equations (UDEs) - ODEs with neural network components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.ScientificML`

Implements Universal Differential Equations (UDEs) - ODEs with neural network components.

## How It Works

For Beginners:
Universal Differential Equations combine known physics with machine learning.

Traditional ODEs:
dx/dt = f(x, t, θ) where f is a known function with parameters θ
Example: dx/dt = -kx (exponential decay, k is known)

Pure Neural ODEs:
dx/dt = NN(x, t, θ) where NN is a neural network

- Very flexible, can learn any dynamics
- But ignores known physics
- May violate physical laws

Universal Differential Equations (UDEs):
dx/dt = f_known(x, t) + NN(x, t, θ)

- Combines known physics (f_known) with learned corrections (NN)
- Best of both worlds!

Key Idea:
Use neural networks to model UNKNOWN parts of the physics while keeping
KNOWN parts as explicit equations.

Example - Epidemic Model:
Known: dS/dt = -βSI, dI/dt = βSI - γI (basic SIR model)
Unknown: How β (infection rate) varies with temperature, policy, etc.
UDE: dS/dt = -β(T, P)SI where β(T, P) = NN(temperature, policy)

Applications:

- Climate modeling (known physics + unknown feedback loops)
- Epidemiology (known disease spread + unknown interventions)
- Chemistry (known reactions + unknown catalysis effects)
- Biology (known population dynamics + unknown environmental factors)
- Engineering (known mechanics + unknown friction/damping)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniversalDifferentialEquation(NeuralNetworkArchitecture<>,Int32,Func<[],,[]>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,UniversalDifferentialEquationsOptions)` | Initializes a new instance of the UniversalDifferentialEquation with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this model supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDerivative([],)` | Computes dx/dt = f_known(x, t) + NN(x, t). |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes UDE-specific data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about the UDE model. |
| `GetOptions` |  |
| `PredictCore(Tensor<>)` | Makes a prediction using the UDE model. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes UDE-specific data. |
| `Simulate([],,,Int32,OdeIntegrationMethod)` | Simulates the UDE forward in time. |
| `Train(Tensor<>,Tensor<>)` | Performs a supervised training step against derivative targets. |
| `UpdateParameters(Vector<>)` | Updates the network parameters from a flattened vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultStateDim` | Initializes a new instance with default settings. |

