---
title: "FinDiffGenerator<T>"
description: "FinDiff generator for synthesizing realistic financial tabular data using diffusion models with temporal correlation preservation and financial constraint enforcement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

FinDiff generator for synthesizing realistic financial tabular data using diffusion
models with temporal correlation preservation and financial constraint enforcement.

## For Beginners

FinDiff generates fake financial data that respects financial rules:

1. Stock prices don't jump randomly — they follow trends (temporal correlation)
2. Prices are always positive (constraint enforcement)
3. Volatile periods cluster together (volatility awareness)

If you provide custom layers in the architecture, those will be used directly
for the denoiser MLP. If not, the network creates industry-standard
FinDiff layers based on the original research paper specifications.

Example usage:

## How It Works

FinDiff augments standard diffusion with financial-specific losses:

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Diffusion Models for Financial Tabular Data" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinDiffGenerator` | Initializes a new FinDiff generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the FinDiff-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the FinDiff network based on the provided architecture. |
| `PredictCore(Tensor<>)` |  |
| `RebuildLayersForDataWidth` | Rebuilds MLP layers using the actual transformed data width (which may differ from Architecture.OutputSize due to VGMM mode-encoding of continuous columns). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

