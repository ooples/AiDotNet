---
title: "TabFlowGenerator<T>"
description: "TabFlow generator using flow matching with optimal transport conditional paths for high-quality, fast synthetic tabular data generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TabFlow generator using flow matching with optimal transport conditional paths
for high-quality, fast synthetic tabular data generation.

## For Beginners

TabFlow works by learning "which direction data should move":

If you provide custom layers in the architecture, those will be used directly
for the velocity field MLP. If not, the network creates industry-standard
TabFlow layers based on the original research paper specifications.

Example usage:

## How It Works

TabFlow learns a velocity field v(x, t) that defines an ODE: dx/dt = v(x, t).
The ODE transports samples from noise (t=0) to data (t=1) along straight paths.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

**Training**: Given data x1 and noise x0, the optimal transport path is:
xt = (1 - t) * x0 + t * x1 (linear interpolation)
Target velocity: v* = x1 - x0 (direction from noise to data)
Loss: ||v(xt, t) - v*||^2 (learn to predict the direction)

**Generation**: Start at x0 ~ N(0,1), solve ODE from t=0 to t=1 using Euler/RK4.

Reference: "Flow Matching for Tabular Data" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabFlowGenerator` | Initializes a new TabFlow generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TabFlow-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the TabFlow generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the TabFlow network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the TabFlow velocity field for the given input tensor. |
| `RebuildLayersForDataWidth` | Rebuilds MLP layers using the actual transformed data width (which may differ from Architecture.OutputSize due to VGMM mode-encoding of continuous columns). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the TabFlow network using the provided input and expected output. |
| `UpdateNetworkParameters` | Updates the parameters of all layers in the network based on computed gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

