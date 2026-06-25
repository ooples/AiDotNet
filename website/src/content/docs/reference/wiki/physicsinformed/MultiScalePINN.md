---
title: "MultiScalePINN<T>"
description: "Implements a Multi-Scale Physics-Informed Neural Network for solving PDEs with multiple scales."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Implements a Multi-Scale Physics-Informed Neural Network for solving PDEs with multiple scales.

## How It Works

For Beginners:
Multi-scale problems are challenging because features span vastly different sizes.
A single neural network struggles to capture both large-scale trends and fine details.

Solution: Multi-Scale PINN
Uses multiple sub-networks, each specialized for a different scale:

- Coarse network: Captures large-scale, smooth variations
- Fine network(s): Capture small-scale details and fluctuations

Architecture:
Input (x,t) → [Coarse Net] → u_coarse(x,t)
→ [Fine Net 1] → u_fine1(x,t)
→ [Fine Net 2] → u_fine2(x,t)
→ ...

Total solution: u(x,t) = u_coarse + u_fine1 + u_fine2 + ...

Training Strategy:

1. Progressive Training: Train coarse first, then add finer scales
2. Simultaneous Training: Train all scales together with adaptive weights
3. Alternating Training: Alternate between scales during training

Key Features:

- Fourier feature encoding at different frequencies for each scale
- Adaptive loss weighting to balance scale contributions
- Scale coupling terms to ensure consistency
- Progressive activation of finer scales during training

Applications:

- Turbulence modeling (large eddies + small vortices)
- Composite materials (macroscopic + fiber-scale behavior)
- Multi-physics problems (thermal + mechanical + chemical)
- Climate modeling (global + regional + local scales)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiScalePINN(NeuralNetworkArchitecture<>,IMultiScalePDE<>,IBoundaryCondition<>[],IInitialCondition<>,Int32,MultiScaleTrainingOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,MultiScalePINNOptions)` | Initializes a new instance of the Multi-Scale PINN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfScales` | Gets the number of scales in this multi-scale PINN. |
| `ParameterCount` |  |
| `ScaleCharacteristicLengths` | Gets the characteristic length scales. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDerivativesForPoint(Tensor<>,Int32,Int32)` | Computes derivatives for a single point using finite differences. |
| `ComputeMultiScaleLoss(Tensor<>)` | Computes the multi-scale physics-informed loss. |
| `CreateNewInstance` |  |
| `CreateScaleArchitecture(NeuralNetworkArchitecture<>,Int32,Int32)` | Creates architecture for a specific scale. |
| `CreateScaleNetworks(NeuralNetworkArchitecture<>)` | Creates the neural networks for each scale. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtractOutputPoint(Tensor<>,Int32)` | Extracts output values for a single point. |
| `ExtractPoint(Tensor<>,Int32)` | Extracts a single point from the input tensor. |
| `Forward(Tensor<>)` | Forward pass through all scale networks. |
| `ForwardWithMemory(Tensor<>)` |  |
| `GenerateCollocationPoints` | Generates collocation points for each scale. |
| `GetGradients` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Solve(Int32,Double,Boolean)` | Solves the multi-scale PDE using physics-informed training. |
| `SolveSequential(Int32,TrainingHistory<>,Boolean)` | Sequential training: train coarse scale first, then progressively add finer scales. |
| `SolveSimultaneous(Int32,TrainingHistory<>,Boolean)` | Simultaneous training: train all scales together with adaptive weighting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainEpoch` | Performs one training epoch. |
| `UpdateAdaptiveScaleWeights` | Updates scale weights adaptively based on gradient magnitudes. |
| `UpdateAllScaleNetworks` | Updates all scale network parameters. |
| `UpdateParameters(Vector<>)` |  |

