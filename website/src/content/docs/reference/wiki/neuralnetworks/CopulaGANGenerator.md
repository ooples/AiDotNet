---
title: "CopulaGANGenerator<T>"
description: "CopulaGAN generator for synthetic tabular data, combining Gaussian copula transformations with the CTGAN training pipeline for improved continuous column modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

CopulaGAN generator for synthetic tabular data, combining Gaussian copula transformations
with the CTGAN training pipeline for improved continuous column modeling.

## For Beginners

CopulaGAN is CTGAN with an extra "normalizing" step:

**Training:****Generation:**

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
CopulaGAN layers based on the original research paper specifications.

Example usage:

## How It Works

CopulaGAN applies a two-stage preprocessing to continuous columns:

1. **Gaussian copula transform**: CDF then quantile then standard normal
2. **VGM normalization**: Standard CTGAN mode-specific normalization

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Synthesizing Tabular Data using Copulas" (2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CopulaGANGenerator` | Initializes a new CopulaGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the CopulaGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCopulaTransform(Matrix<>)` | Applies Gaussian copula transform to continuous columns: value -> empirical CDF -> inverse normal CDF (quantile function). |
| `ApplyInverseCopulaTransform(Matrix<>)` | Applies inverse Gaussian copula transform to continuous columns. |
| `ApplyOutputActivations(Tensor<>)` | Applies tanh to continuous columns and softmax to categorical/mode columns. |
| `ApplyOutputActivationsBatched(Tensor<>)` | Batched per-column VGM output activation: Tanh on continuous mode value, Softmax on mode probabilities, Softmax on categorical blocks. |
| `BuildDiscriminatorDimensionMap(Int32)` | Builds a dimension map for the discriminator layers to support manual backward pass. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForward(Tensor<>,Boolean)` | Runs the discriminator forward pass. |
| `DiscriminatorForwardBatched(Tensor<>,Boolean)` | Batched, tape-tracked discriminator forward (LeakyReLU 0.2 + Dropout). |
| `Erf(Double)` | Approximation of the error function using Abramowitz and Stegun formula 7.1.26. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the CopulaGAN generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |
| `GeneratorForwardWithResidual(Tensor<>)` | Generator forward pass with CTGAN-style residual connections: Each hidden layer receives [previous_output, original_input]. |
| `GeneratorForwardWithResidualBatched(Tensor<>)` | Batched, tape-tracked generator forward with residual skip connections. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the CopulaGAN network based on the provided architecture. |
| `NormalCDF(Double)` | Approximation of the normal CDF using the error function. |
| `NormalQuantile(Double)` | Approximation of the inverse normal CDF using Beasley-Springer-Moro algorithm. |
| `PredictCore(Tensor<>)` | Runs the generator forward pass to create synthetic data from a noise input. |
| `RebuildLayersWithActualDimensions(Int32,Int32,Int32)` | Rebuilds the default generator and discriminator layers with actual data dimensions discovered during Fit(). |
| `SafeGradient(Tensor<>,Double)` | Applies NaN sanitization and gradient norm clipping in a single operation. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the CopulaGAN network using the provided input and expected output. |
| `TrainDiscriminatorStepBatched(Matrix<>,Int32)` | Paper-faithful WGAN-GP critic step (Xu et al. |
| `TrainGeneratorStepBatched(Matrix<>,Int32)` | Paper-faithful WGAN-GP generator step. |
| `UpdateDiscriminatorParameters()` | Updates discriminator parameters with a given learning rate. |
| `UpdateGeneratorParameters()` | Updates generator parameters with a given learning rate. |
| `UpdateParameters(Vector<>)` |  |

