---
title: "InverseProblemPINN<T>"
description: "Implements a Physics-Informed Neural Network for inverse problems (parameter identification)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Implements a Physics-Informed Neural Network for inverse problems (parameter identification).

## How It Works

For Beginners:
Inverse problems are about discovering unknown parameters from observations.
This is the opposite of forward problems where parameters are known.

Examples of Inverse Problems:

1. Medical Imaging: Find tumor location from external measurements
2. Material Science: Identify Young's modulus from stress-strain data
3. Geophysics: Determine subsurface properties from seismic data
4. Finance: Calibrate model parameters from market prices

How InverseProblemPINN Works:

1. Neural network learns the solution u(x,t)
2. Additional trainable variables represent unknown parameters θ
3. Both are trained together to minimize:
- Data loss: ||u_predicted - u_observed||²
- Physics loss: ||PDE_residual(u, θ)||²
- Regularization: Prior knowledge about parameters

Key Advantages:

- Handles noisy and sparse observations
- Physics acts as regularization
- No need for iterative PDE solves
- Can quantify parameter uncertainty

Training Strategy:

1. Initialize parameters near prior estimates
2. Train with emphasis on physics initially
3. Gradually increase data weight
4. Use separate learning rates for network and parameters

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InverseProblemPINN(NeuralNetworkArchitecture<>,IInverseProblem<>,IBoundaryCondition<>[],IInitialCondition<>,Int32,InverseProblemOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Initializes a new instance of the InverseProblemPINN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `ParameterHistory` | Gets the parameter estimation history. |
| `ParameterNames` | Gets the parameter names. |
| `Parameters` | Gets the current estimated parameter values. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateAndUpdate(,,)` | Backpropagates losses and updates network parameters. |
| `ComputeDataLoss` | Computes the data loss (fit to observations). |
| `ComputeDerivatives(Tensor<>)` | Computes derivatives using finite differences. |
| `ComputeParameterGradients` | Computes gradients of the loss with respect to parameters using finite differences. |
| `ComputePhysicsLoss` | Computes the physics loss (PDE residual). |
| `ComputeRegularizationLoss` | Computes the regularization loss for parameters. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EstimateParameterCorrelations` | Estimates correlation matrix between parameters. |
| `EstimateParameterUncertainties` | Estimates parameter uncertainties using Fisher information. |
| `ForwardWithMemory(Tensor<>)` |  |
| `GenerateCollocationPoints` | Generates collocation points for PDE enforcement. |
| `GetGradients` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Solve(Int32,Double,Boolean)` | Solves the inverse problem to identify unknown parameters. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainEpoch` | Performs one training epoch. |
| `UpdatePDE` | Updates the PDE with current parameter values. |
| `UpdateParameters(Vector<>)` |  |
| `UpdateUnknownParameters` | Updates the unknown parameters using gradient descent. |

