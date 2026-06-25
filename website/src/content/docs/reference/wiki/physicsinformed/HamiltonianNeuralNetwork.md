---
title: "HamiltonianNeuralNetwork<T>"
description: "Implements Hamiltonian Neural Networks (HNN) for learning conservative dynamical systems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.ScientificML`

Implements Hamiltonian Neural Networks (HNN) for learning conservative dynamical systems.

## How It Works

For Beginners:
Hamiltonian Neural Networks learn the laws of physics by respecting conservation principles.

Classical Mechanics - Hamiltonian Formulation:
Many physical systems are described by Hamilton's equations:

- dq/dt = ∂H/∂p (position changes with momentum gradient)
- dp/dt = -∂H/∂q (momentum changes with negative position gradient)

Where:

- q = position coordinates
- p = momentum coordinates
- H(q,p) = Hamiltonian (total energy of the system)

Key Property: Energy Conservation
For conservative systems, H(q,p) = constant (energy is conserved)

Traditional Neural Networks vs. HNN:

Traditional NN:

- Learn dynamics directly: (q,p) → (dq/dt, dp/dt)
- Can violate physics laws
- May not conserve energy
- Can accumulate errors over time

HNN:

- Learn the Hamiltonian: (q,p) → H
- Compute dynamics from H: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
- Automatically conserves energy (by construction!)
- More accurate long-term predictions

How It Works:

1. Neural network learns H(q,p)
2. Use automatic differentiation to get ∂H/∂q and ∂H/∂p
3. Apply Hamilton's equations to get dynamics
4. Guaranteed to preserve Hamiltonian structure

Applications:

- Planetary motion (gravitational systems)
- Molecular dynamics (particle interactions)
- Robotics (mechanical systems)
- Quantum mechanics (Schrödinger equation)
- Any conservative system

Example - Pendulum:
H(q,p) = p²/(2m) + mgl(1 - cos(q))

- q = angle, p = angular momentum
- HNN learns this from data without knowing the formula!

Key Benefit:
By encoding physics structure (Hamiltonian formulation), the network
learns faster, generalizes better, and makes physically consistent predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this model supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeHamiltonian([])` | Computes the Hamiltonian (energy) for a given state. |
| `ComputeTimeDerivative([])` | Computes the time derivative of the state using Hamilton's equations. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Hamiltonian-specific data. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about the Hamiltonian network. |
| `GetOptions` |  |
| `PredictCore(Tensor<>)` | Makes a prediction using the Hamiltonian network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Hamiltonian-specific data. |
| `Simulate([],,Int32)` | Simulates the system forward in time. |
| `Train(Tensor<>,Tensor<>)` | Trains the Hamiltonian neural network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the network parameters from a flattened vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultStateDim` | Initializes a new instance with default settings. |

