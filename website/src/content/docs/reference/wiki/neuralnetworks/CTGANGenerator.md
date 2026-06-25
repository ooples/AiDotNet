---
title: "CTGANGenerator<T>"
description: "Conditional Tabular GAN (CTGAN) for generating realistic synthetic tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Conditional Tabular GAN (CTGAN) for generating realistic synthetic tabular data.

## For Beginners

CTGAN works like a forgery competition:

1. The **Generator** starts with random noise and tries to create realistic table rows
2. The **Discriminator** sees both real and generated rows and tries to tell them apart
3. They train together: the generator gets better at fooling the discriminator,

and the discriminator gets better at spotting fakes

4. Eventually, the generator produces rows that are indistinguishable from real data

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
CTGAN layers based on the original research paper specifications.

Example usage:

## How It Works

CTGAN combines several innovations for effective tabular data generation:

- **VGM normalization**: Handles multi-modal continuous distributions
- **Conditional generation**: Training-by-sampling ensures all categories are represented
- **WGAN-GP loss**: Wasserstein distance with gradient penalty for stable training
- **Residual generator**: Skip connections in the generator for better gradient flow
- **PacGAN**: Packing multiple samples to prevent mode collapse

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CTGANGenerator` | Initializes a new CTGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the CTGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOutputActivationsBatched(Tensor<>)` | Batched per-column output activations (Xu et al. |
| `BuildCategoricalOutputBlocks` | Paper-faithful WGAN-GP generator step (Xu et al. |
| `BuildDiscriminatorDimensionMap(Int32)` | Builds a dimension map for the discriminator layers to support manual backward pass. |
| `BuildPackedRealAndFakeBatches(Matrix<>,Int32)` | Builds the packed real + fake batches for one critic step. |
| `ConcatInto(Vector<>,Vector<>,Vector<>)` | Concatenates two vectors into a pre-allocated destination buffer. |
| `ConditionalCrossEntropy(Tensor<>,Tensor<>,Tensor<>)` | Conditional generator loss term (Xu 2019 §4.3): cross-entropy between the conditional one-hot and the generator's output distribution for the conditioned discrete column, masked to that column. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForward(Tensor<>,Boolean)` | Discriminator forward pass with LeakyReLU and dropout. |
| `DiscriminatorForwardBatched(Tensor<>,Boolean)` | Batched, tape-tracked version of `Boolean)`. |
| `FillFromTensor(Tensor<>,Vector<>)` | Fills a pre-allocated vector from a tensor. |
| `FillRow(Matrix<>,Int32,Vector<>)` | Fills a pre-allocated vector with a row from the data matrix. |
| `FillStandardNormal(Vector<>)` | Fills a pre-allocated vector with standard normal random values (Box-Muller). |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the CTGAN generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |
| `GenerateNoiseBatchTensor(Int32)` | Batched Box-Muller standard-normal noise generator (matches `Int32)`). |
| `GeneratorForwardWithResidual(Tensor<>)` | Generator forward pass with residual connections: each hidden layer receives both the previous output and the original input concatenated. |
| `GeneratorForwardWithResidualBatched(Tensor<>)` | Batched, tape-tracked version of `Tensor{`. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `InitializeTrainingBuffers(Int32)` | Pre-allocates reusable buffers for the training loop to eliminate per-row GC pressure. |
| `PredictCore(Tensor<>)` |  |
| `RebuildLayersWithActualDimensions(Int32,Int32,Int32)` | Rebuilds generator and discriminator layers with actual data dimensions discovered during Fit(). |
| `SampleCondMaskBatch(Int32)` | Samples a batch of conditional vectors together with their masks (Xu 2019 §4.3). |
| `SampleConditionalBatchTensor(Int32)` | Samples a batch of conditional vectors from the CTGAN sampler (one per sample). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatched(Matrix<>,Int32)` | Paper-faithful WGAN-GP critic step (Xu et al. |
| `UpdateParameters(Vector<>)` |  |

