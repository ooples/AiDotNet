---
title: "TableGANGenerator<T>"
description: "TableGAN generator using a DCGAN-style architecture with classification and information loss for high-quality synthetic tabular data generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TableGAN generator using a DCGAN-style architecture with classification and information
loss for high-quality synthetic tabular data generation.

## For Beginners

TableGAN is like a regular GAN with two extra quality checks:

1. **Adversarial loss**: Does the synthetic data look real? (WGAN-GP for stable training)
2. **Classification loss**: Are label-feature relationships preserved?
3. **Information loss**: Do the mean/variance statistics match the real data?

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
TableGAN layers based on the original research paper specifications.

Example usage:

## How It Works

TableGAN optimizes three losses simultaneously:

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full autodiff and JIT compilation support

Reference: "Data Synthesis based on Generative Adversarial Networks" (Park et al., 2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TableGANGenerator` | Initializes a new instance with default architecture settings. |
| `TableGANGenerator(NeuralNetworkArchitecture<>,TableGANOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new TableGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TableGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOutputActivationsBatched(Tensor<>)` | Batched output activations (Park et al. |
| `BuildRealBatchTensor(Matrix<>,Int32)` | Samples `batchSize` rows uniformly at random from the post-transformer matrix into a rank-2 tensor `[B, dataWidth]`. |
| `ComputeGradientPenalty(Tensor<>,Tensor<>)` | WGAN-GP gradient penalty (Gulrajani et al. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForwardBatched(Tensor<>,Boolean)` | Batched discriminator forward, fully tape-tracked. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GenerateNoiseBatchTensor(Int32)` | Generates a batched Gaussian noise tensor `[B, embeddingDim]` via vectorized Box-Muller using `TensorRandomUniformRange`. |
| `GeneratorForwardBatched(Tensor<>)` | Batched generator forward, fully tape-tracked. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatched(Tensor<>,Tensor<>)` | Paper-faithful Wasserstein critic update (Park et al. |
| `TrainGeneratorStepBatched(Tensor<>,Tensor<>)` | Paper-faithful generator update (Park et al. |
| `UpdateParameters(Vector<>)` |  |

