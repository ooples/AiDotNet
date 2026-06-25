---
title: "AutoDiffTabGenerator<T>"
description: "AutoDiff-Tab generator that automatically searches over diffusion configurations (timesteps, noise schedules, network architecture) to find optimal settings for tabular data generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

AutoDiff-Tab generator that automatically searches over diffusion configurations
(timesteps, noise schedules, network architecture) to find optimal settings for
tabular data generation.

## For Beginners

AutoDiff-Tab is a "smart" version of TabDDPM:

1. It tries several different diffusion setups (like testing different recipes)
2. Evaluates which one works best on your data
3. Uses the winner for the final model

This saves you from having to manually tune hyperparameters.

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

If you provide custom layers in the architecture, those will be used directly
for the denoiser network. Otherwise, the network creates the standard architecture
based on the best configuration found during search.

Example usage:

## How It Works

AutoDiff-Tab combines architecture search with diffusion models:

Reference: "Automated Diffusion Models for Tabular Data" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoDiffTabGenerator` | Initializes a new AutoDiff-Tab generator with the specified architecture. |
| `AutoDiffTabGenerator(NeuralNetworkArchitecture<>,AutoDiffTabOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new instance of the AutoDiffTabGenerator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the AutoDiffTab-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDenoiserInput(Vector<>,Int32)` | Concatenates a data row with the (projected) timestep embedding into the denoiser input vector. |
| `CreateNewInstance` |  |
| `DenoiserForward(Tensor<>)` | Shared denoiser forward over the hidden MLP (+ dropout) and output head, taking a [dataWidth + timestepEmbeddingDim] augmented input. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EnsureSizedForInput(Tensor<>)` | When unfitted (the generated ModelFamily tests call Train/Predict without Fit), adapt the denoiser to the actual input width so its first layer accepts width + timestep-embedding inputs. |
| `ExtractLayerReferences` | Extracts private layer references from the unified Layers list. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the AutoDiff-Tab generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `ForwardForTraining(Tensor<>)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows using the trained diffusion model. |
| `GetFeatureImportance` |  |
| `GetLayerOutputSize(ILayer<>)` | Estimates the output size of a layer by its parameter count. |
| `GetModelMetadata` |  |
| `PredictCore(Tensor<>)` | Runs the denoiser forward pass to predict noise from noisy data + timestep embedding. |
| `RebuildDenoiserLayers(Int32[],Int32,Int32)` | Rebuilds the denoiser layers with actual data dimensions discovered during Fit(). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TryGetArchitectureInputShape` | Initializes the layers of the AutoDiff-Tab denoiser based on the provided architecture. |
| `UpdateParameters(Vector<>)` |  |

