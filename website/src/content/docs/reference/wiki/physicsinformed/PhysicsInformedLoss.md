---
title: "PhysicsInformedLoss<T>"
description: "Loss function for Physics-Informed Neural Networks (PINNs)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Loss function for Physics-Informed Neural Networks (PINNs).
Combines data loss, PDE residual loss, boundary condition loss, and initial condition loss.

## How It Works

For Beginners:
Traditional neural networks learn from data alone. Physics-Informed Neural Networks (PINNs)
additionally enforce that the solution satisfies physical laws (PDEs) and constraints.

This loss function has multiple components:

1. Data Loss: Measures how well predictions match observed data points
2. PDE Residual Loss: Measures how much the PDE is violated at collocation points
3. Boundary Loss: Ensures boundary conditions are satisfied
4. Initial Condition Loss: Ensures initial conditions are satisfied (for time-dependent problems)

The total loss is a weighted sum:
L_total = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic

Why This Works:
By minimizing this loss, the network learns to:

- Fit the available data
- Obey physical laws everywhere (not just at data points)
- Satisfy boundary and initial conditions

This often requires far less data than traditional deep learning!

Key Innovation:
PINNs can solve PDEs in regions where we have NO data, as long as the physics is known.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PhysicsInformedLoss(IPDESpecification<>,IBoundaryCondition<>[],IInitialCondition<>,Nullable<Double>,Nullable<Double>,Nullable<Double>,Nullable<Double>)` | Initializes a new instance of the Physics-Informed loss function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the loss function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDerivative(Vector<>,Vector<>)` |  |
| `CalculateLoss(Vector<>,Vector<>)` |  |
| `ComputeBoundaryLoss(Vector<>,Vector<>,PDEDerivatives<>)` | Computes the boundary condition loss. |
| `ComputeDataLoss(Vector<>,Vector<>)` | Computes the data fitting loss (Mean Squared Error). |
| `ComputeDerivative([],[])` | Computes the derivative of the loss with respect to predictions. |
| `ComputeInitialLoss(Vector<>,Vector<>)` | Computes the initial condition loss. |
| `ComputeLoss(Vector<>,Vector<>,PDEDerivatives<>,Vector<>)` | Computes the total physics-informed loss (compatibility wrapper). |
| `ComputePDELoss(Vector<>,Vector<>,PDEDerivatives<>)` | Computes the PDE residual loss. |
| `ComputePhysicsLoss(Vector<>,Vector<>,PDEDerivatives<>,Vector<>)` | Computes the total physics-informed loss for PINN training. |
| `ComputePhysicsLossGradients(Vector<>,Vector<>,PDEDerivatives<>,Vector<>)` | Computes the physics-informed loss and its gradients with respect to outputs and derivatives. |
| `ComputeTapeLoss(Tensor<>,Tensor<>)` |  |
| `SetCollocationCoordinates(Tensor<>)` | Sets the collocation points where the PDE residual will be evaluated. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_collocationCoordinates` | Optional collocation coordinates for PDE residual evaluation. |

