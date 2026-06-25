---
title: "PhysicsInformedNeuralNetwork<T>"
description: "Represents a Physics-Informed Neural Network (PINN) for solving PDEs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Represents a Physics-Informed Neural Network (PINN) for solving PDEs.

## How It Works

For Beginners:
A Physics-Informed Neural Network (PINN) is a neural network that learns to solve
Partial Differential Equations (PDEs) by incorporating physical laws directly into
the training process.

Traditional Approach (Finite Elements/Differences):

- Discretize the domain into a grid
- Approximate derivatives using neighboring points
- Solve a large system of equations
- Works well but can be slow for complex geometries

PINN Approach:

- Neural network represents the solution u(x,t)
- Use automatic differentiation to compute ∂u/∂x, ∂²u/∂x², etc.
- Train the network to minimize:
* PDE residual (how much the PDE is violated)
* Boundary condition errors
* Initial condition errors
* Data fitting errors (if measurements are available)

Key Advantages:

1. Meshless: No need to discretize the domain
2. Data-efficient: Can work with sparse or noisy data
3. Flexible: Easy to handle complex geometries and boundary conditions
4. Interpolation: Get solution at any point by evaluating the network
5. Inverse problems: Can discover unknown parameters in the PDE

Key Challenges:

1. Training can be difficult (multiple objectives to balance)
2. May require careful tuning of loss weights
3. Network architecture affects accuracy
4. Computational cost during training (many derivative evaluations)

Applications:

- Fluid dynamics (Navier-Stokes equations)
- Heat transfer
- Structural mechanics
- Quantum mechanics
- Financial modeling (Black-Scholes PDE)
- Climate and weather modeling

Historical Context:
PINNs were introduced by Raissi, Perdikaris, and Karniadakis in 2019.
They've revolutionized scientific machine learning by showing that deep learning
can be guided by physics rather than just data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PhysicsInformedNeuralNetwork(NeuralNetworkArchitecture<>,IPDESpecification<>,IBoundaryCondition<>[],IInitialCondition<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Nullable<Double>,Nullable<Double>,Nullable<Double>,Nullable<Double>,PhysicsInformedNeuralNetworkOptions)` | Initializes a new instance of the PINN class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether this PINN supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes PINN-specific data. |
| `EvaluateAtPoint([])` | Evaluates the network at a single point. |
| `EvaluatePDEResidual([])` | Evaluates the PDE residual at a point (for validation). |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GenerateCollocationPoints` | Generates collocation points for enforcing the PDE in the domain. |
| `GetModelMetadata` | Gets metadata about the PINN model. |
| `GetOptions` |  |
| `GetSolution([])` | Gets the solution at a specific point in the domain. |
| `InitializeLayers` | Initializes the neural network layers. |
| `PredictCore(Tensor<>)` | Makes a prediction using the PINN. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes PINN-specific data. |
| `SetCollocationPoints([0:,0:])` | Sets custom collocation points (for advanced users who want specific sampling). |
| `SumDerivatives(PDEDerivatives<>,PDEDerivatives<>)` | Sums two sets of PDE derivatives element-wise. |
| `Train(Tensor<>,Tensor<>)` | Performs a basic supervised training step using MSE loss. |
| `UpdateParameters(Vector<>)` | Updates the network parameters from a flattened vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_pdeSpecification` | The PDE specification that defines the physics constraints. |

