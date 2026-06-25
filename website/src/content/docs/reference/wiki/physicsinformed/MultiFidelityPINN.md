---
title: "MultiFidelityPINN<T>"
description: "Multi-fidelity Physics-Informed Neural Network for combining data of different accuracy levels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Multi-fidelity Physics-Informed Neural Network for combining data of different accuracy levels.

## How It Works

For Beginners:
Multi-fidelity learning combines data from multiple sources with different accuracy levels:

Low-Fidelity Data (Cheap, Abundant):

- Coarse simulations
- Simplified physical models
- Fast but approximate calculations
- Example: 2D simulation of a 3D problem

High-Fidelity Data (Expensive, Scarce):

- Fine-mesh simulations
- Physical experiments
- High-accuracy calculations
- Example: Wind tunnel measurements

The Multi-Fidelity Approach:

1. Train on abundant low-fidelity data to learn general trends
2. Use scarce high-fidelity data to correct errors
3. Learn the correlation between fidelity levels
4. Enforce physics constraints at all fidelity levels

Mathematical Model:
u_HF(x) = rho(x) * u_LF(x) + delta(x)

Where:

- u_LF(x): Low-fidelity prediction
- u_HF(x): High-fidelity prediction
- rho(x): Scaling factor (learned)
- delta(x): Correction/bias term (learned)

This implementation uses a nonlinear correlation model where a neural network
learns the relationship between fidelity levels.

References:

- Meng, X., and Karniadakis, G.E. "A composite neural network that learns from

multi-fidelity data: Application to function approximation and inverse PDE problems"
Journal of Computational Physics, 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiFidelityPINN(NeuralNetworkArchitecture<>,IPDESpecification<>,IBoundaryCondition<>[],IInitialCondition<>,PhysicsInformedNeuralNetwork<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Double,Double,Double,Double,Double,Boolean,MultiFidelityPINNOptions)` | Creates a Multi-Fidelity PINN with optional custom low-fidelity network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLowFidelityFrozen` | Gets whether the low-fidelity network is frozen. |
| `LowFidelityNetwork` | Gets the low-fidelity network for external access. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePhysicsLossAtPoints` | Computes the physics loss by evaluating PDE residual at collocation points. |
| `CreateLowFidelityArchitecture(NeuralNetworkArchitecture<>,IPDESpecification<>)` | Creates a smaller architecture for the low-fidelity network. |
| `GetFidelityCorrection([])` | Gets the correction (difference between fidelity levels) at a point. |
| `GetHighFidelitySolution([])` | Gets the high-fidelity prediction at a point. |
| `GetLowFidelitySolution([])` | Gets the low-fidelity prediction at a point. |
| `GetOptions` |  |
| `SetHighFidelityData(Tensor<>,Tensor<>)` | Sets the high-fidelity training data. |
| `SetLowFidelityData(Tensor<>,Tensor<>)` | Sets the low-fidelity training data. |
| `SetLowFidelityFrozen(Boolean)` | Freezes or unfreezes the low-fidelity network. |
| `SolveMultiFidelity(Int32,Nullable<Int32>,Double,Boolean,Int32)` | Solves the PDE using multi-fidelity training. |

