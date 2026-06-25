---
title: "LagrangianNeuralNetwork<T>"
description: "Implements Lagrangian Neural Networks (LNN) for learning mechanical systems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.ScientificML`

Implements Lagrangian Neural Networks (LNN) for learning mechanical systems.

## How It Works

For Beginners:
Lagrangian Neural Networks learn physics using the Lagrangian formulation of mechanics.

Lagrangian Mechanics:
Alternative to Hamiltonian mechanics, uses the Lagrangian:
L(q, q̇) = T - V = Kinetic Energy - Potential Energy

Equations of Motion (Euler-Lagrange):
d/dt(∂L/∂q̇) - ∂L/∂q = 0

Where:

- q = generalized coordinates (positions)
- q̇ = generalized velocities
- T = kinetic energy (usually ½m q̇²)
- V = potential energy (depends on q)

Why Lagrangian vs. Hamiltonian?

- Lagrangian: Uses (q, q̇) - position and velocity
- Hamiltonian: Uses (q, p) - position and momentum
- Lagrangian often more intuitive for mechanical systems
- Both give same physics, different formulations

How LNN Works:

1. Neural network learns L(q, q̇)
2. Compute ∂L/∂q and ∂L/∂q̇ using automatic differentiation
3. Apply Euler-Lagrange equation to get acceleration q̈
4. Guaranteed to conserve energy and satisfy principle of least action

Applications:

- Robotics (manipulator dynamics)
- Biomechanics (human motion)
- Aerospace (satellite dynamics)
- Any mechanical system

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LagrangianNeuralNetwork` | Creates a default LagrangianNeuralNetwork with a 2D configuration space. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this model supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAcceleration([],[])` | Computes acceleration using Euler-Lagrange equation. |
| `ComputeLagrangian([],[])` | Computes the Lagrangian L(q, q̇) = T - V. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Lagrangian-specific data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about the Lagrangian network. |
| `GetOptions` |  |
| `PredictCore(Tensor<>)` | Makes a prediction using the Lagrangian network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Lagrangian-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the Lagrangian neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the network parameters from a flattened vector. |

