---
title: "VariationalPINN<T>"
description: "Implements Variational Physics-Informed Neural Networks (VPINNs)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Implements Variational Physics-Informed Neural Networks (VPINNs).

## How It Works

For Beginners:
Variational PINNs (VPINNs) use the weak (variational) formulation of PDEs instead of
the strong form. This is similar to finite element methods (FEM).

Strong vs. Weak Formulation:

Strong Form (standard PINN):

- PDE must hold pointwise: PDE(u) = 0 at every point
- Example: -∇²u = f everywhere
- Requires computing second derivatives
- Solution must be twice differentiable

Weak Form (VPINN):

- PDE holds "on average" against test functions
- ∫∇u·∇v dx = ∫fv dx for all test functions v
- Integration by parts reduces derivative order
- Solution only needs to be once differentiable
- More stable numerically

Key Advantages:

1. Lower derivative requirements (better numerical stability)
2. Natural incorporation of boundary conditions (through integration by parts)
3. Can handle discontinuities and rough solutions better
4. Closer to FEM (well-understood mathematical theory)
5. Often better convergence properties

How VPINNs Work:

1. Choose test functions (often neural networks themselves)
2. Multiply PDE by test function and integrate
3. Use integration by parts to reduce derivative order
4. Minimize the residual in the weak sense

Example - Poisson Equation:
Strong: -∇²u = f
Weak: ∫∇u·∇v dx = ∫fv dx (after integration by parts)

VPINNs train the network u(x) to satisfy the weak form for all test functions v.

Applications:

- Same as PINNs, but particularly useful for:
* Problems with rough solutions
* Conservation laws
* Problems where weak solutions are more natural
* High-order PDEs (where reducing derivative order helps)

Comparison with Standard PINNs:

- VPINN: More stable, lower derivative requirements, closer to FEM
- Standard PINN: Simpler to implement, direct enforcement of PDE

The variational formulation often provides better training dynamics and accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalPINN(NeuralNetworkArchitecture<>,Func<[],[],[0:,0:],[],[0:,0:],>,Int32,Int32,VariationalPINNOptions)` | Initializes a new instance of the Variational PINN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this model supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeTestFunctionGradient([],Int32)` | Computes the gradient of a test function. |
| `ComputeWeakResidual(Int32)` | Computes the weak form residual by integrating over the domain. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes VPINN-specific data. |
| `EvaluateTestFunction([],Int32)` | Evaluates a test function (simple polynomial basis for demonstration). |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GenerateMultiIndex(Int32,Int32)` | Generates a multi-index for polynomial basis. |
| `GetModelMetadata` | Gets metadata about the VPINN model. |
| `GetOptions` |  |
| `GetSolution([])` | Gets the solution at a specific point. |
| `PredictCore(Tensor<>)` | Makes a prediction using the VPINN. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes VPINN-specific data. |
| `Train(Tensor<>,Tensor<>)` | Performs a basic supervised training step using MSE loss. |
| `UpdateParameters(Vector<>)` | Updates the network parameters from a flattened vector. |

