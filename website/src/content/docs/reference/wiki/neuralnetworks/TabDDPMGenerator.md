---
title: "TabDDPMGenerator<T>"
description: "TabDDPM (Tabular Denoising Diffusion Probabilistic Model) for generating synthetic tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TabDDPM (Tabular Denoising Diffusion Probabilistic Model) for generating synthetic tabular data.

## For Beginners

TabDDPM generates data by learning to "undo noise."

**Training:****Generation:**

If you provide custom layers in the architecture, those will be used directly
for the denoiser MLP. If not, the network creates industry-standard
TabDDPM layers based on the original research paper specifications.

Example usage:

TabDDPM often produces higher-quality synthetic data than CTGAN/TVAE, especially
for complex distributions, at the cost of slower generation (many denoising steps).

## How It Works

TabDDPM applies diffusion models to tabular data with separate processes for different feature types:

- **Gaussian diffusion** for continuous/numerical features (noise prediction)
- **Multinomial diffusion** for categorical features (category probability prediction)
- **Shared MLP denoiser** with sinusoidal timestep embedding processes both types jointly
- **Simple preprocessing**: Quantile normalization for continuous, integer encoding for categorical

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabDDPM: Modelling Tabular Data with Diffusion Models" (Kotelnikov et al., ICML 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabDDPMGenerator` | Initializes a new TabDDPM generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TabDDPM-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiffusionLossTape(Tensor<>,Vector<>,Tensor<>,Vector<>)` | Tape-connected TabDDPM loss: numerical ε-MSE + per-group categorical softmax cross-entropy. |
| `CreateNewInstance` |  |
| `CreateTimestepEmbeddingTensor(Int32)` | Tape-connected timestep embedding: the sinusoidal features projected through the learnable `_timestepProjection`, returned as a `Tensor` so its gradient flows back to the projection during training. |
| `DenoiserForwardTensors(Vector<>,Vector<>,Tensor<>)` | Tape-connected denoiser forward returning the numerical ε-prediction and the categorical logits as `Tensor` outputs (the vector- returning `Vector{` is kept for sampling/inference). |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetExtraTrainableTensors` | Exposes the numerical/categorical output-head parameters to the tape-based optimizer step. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `RebuildLayersWithActualDimensions` | Rebuilds denoiser layers with actual data dimensions discovered during Fit(). |
| `ReduceToScalar(Tensor<>)` | Reduces a tensor to a scalar [1] by summing all elements (tape-connected). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SoftmaxCrossEntropy(Tensor<>,Tensor<>,Int32,Int32)` | Tape-connected softmax cross-entropy −Σ target·log(softmax(logit_slice)) over a categorical group. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

